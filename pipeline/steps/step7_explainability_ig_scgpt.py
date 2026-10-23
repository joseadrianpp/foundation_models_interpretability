"""
Integrated Gradients (IG) attribution for the scGPT models, plus what every
model's IG shares (IGResult, the checkpointed IG loop and the CSSI version),
used by step7_explainability_ig_geneformer.py too.

We use Captum's IntegratedGradients (Gauss-Legendre quadrature,
n_steps interpolation points)

Why IG and not attention: attention tells us what the model looks at, not
what drives its decision. IG measures, for every gene, how much that gene's
measured expression pushes the cell towards the PD prediction.

How it works:
  - Captum integrates the gradient of one output along the straight path
    from baseline (every value 0) to the real cell, using `n_steps`
    Gauss-Legendre points. IG(gene) = (real_value - baseline_value) *
    integrated_gradient.
  - The output is the PD logit for the classifiers (zero-shot + MLP head,
    LoRA): IG of their correctly classified cells. The encoders without a
    head (zero-shot, Step 2.5) have no prediction, so they use the output of
    SIGnature (Gold et al., Nat Biotech 2026): sum_k |w_k * f_k(x')|, with
    f the cell embedding and w = f(x) the embedding of the real cell, fixed;
    IG of every cell, each cell's attributions scaled so their absolute
    values sum to 1000 (SIGnature's normalize_attribution_matrix).
  - The signed IG of every gene token is kept (IGResult.token_ig). Score
    `IG`, the original one of each method: the MEAN ABSOLUTE attribution
    over the scored cells that have the gene, if at least
    C.MIN_PRESENT_FRAC of them have it (non-negative, like RF importance;
    SIGnature also takes the absolute value). Its CSSI version
    (stratified_ig_scores) takes, per gene, the maximum of that mean over
    the Leiden strata of the model's own cell embedding.
  - Score `IG±` (to_signed_cache): the signed IG in an AttentionCache, so it
    gets the attention's PD - control mean, CSSI and selections. With the
    PD logit, mean_pd(IG) - mean_control(IG) is the sum of the two
    conditions' mean IG towards the true class, so a gene that pushes each
    condition to its own label scores high instead of cancelling out.

The cells (correctly classified or all of them) and the cell embeddings come
from the model's attention pass (attention_cssi_leiden.collect_scgpt), the
same one its attention task runs, so the two tasks share their Leiden strata.
IG over every cell takes hours on the large cell types,
so the per-token attributions computed so far are saved every
CHECKPOINT_MINUTES (run_ig) and a rerun resumes from them.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from .. import config as C
from ..utils.progress import progress
from .attention_cssi_leiden import AttentionCache

# Gauss-Legendre interpolation points and forward batch size. IG needs a
# backward pass per step, so the batch is smaller than the attention path's.
# 50 steps, as in pipeline_unified, whose IG results are the same score
IG_STEPS = 50
IG_BATCH_SIZE = 8
# Captum runs IG_BATCH_SIZE * IG_STEPS interpolated cells per forward/backward;
# splitting them into chunks of this size keeps an 8 GB GPU from running out of
# memory and does not change the attributions (the gradients are just summed)
IG_INTERNAL_BATCH_SIZE = 32

# Minutes between two saves of the IG computed so far (run_ig)
CHECKPOINT_MINUTES = 15

# SIGnature scales every cell's |attributions| to sum to this
SIGNATURE_TARGET_SUM = 1000.0
# IG outputs: the classifiers' PD logit, or SIGnature's weighted embedding sum
OUTPUTS = ("pd_logit", "signature")


# ---------------------------------------------------------------------------
# Shared by the IG of every model
# ---------------------------------------------------------------------------
def attribute(attributor: IntegratedGradients, internal_batch_size: list, **kwargs) -> torch.Tensor:
    """attributor.attribute, halving the internal batch while the GPU runs out of memory.

    `internal_batch_size` is a one-element list, so the size that fits is kept
    for the next batches. The chunks only split the sum over the steps, so
    the attributions do not change.
    """
    while True:
        try:
            return attributor.attribute(internal_batch_size=internal_batch_size[0], **kwargs)
        except torch.cuda.OutOfMemoryError:
            if internal_batch_size[0] <= 1:
                raise
            torch.cuda.empty_cache()
            internal_batch_size[0] = max(1, internal_batch_size[0] // 2)
            print(f"  IG: out of GPU memory, internal batch -> {internal_batch_size[0]}", flush=True)


def run_ig(order: np.ndarray, make_batches: Callable[[np.ndarray], List[np.ndarray]],
           ig_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
           checkpoint: Optional[Path] = None, desc: str = "IG") -> Tuple[np.ndarray, ...]:
    """IG of the cells in `order`, batched by `make_batches`, saved every CHECKPOINT_MINUTES.

    ig_fn(cell indices) returns the batch's token ids (B, L), signed IG (B, L)
    and a (B, L) mask of the gene tokens to keep. Returns the cell, token id
    and IG of every kept token, cell after cell in `order`. With `checkpoint`,
    the tokens of the cells done so far go to that .npz every
    CHECKPOINT_MINUTES; a rerun whose `order` starts with those cells resumes
    after them (the batches may differ, e.g. on another GPU: each cell's IG
    does not depend on the others). The checkpoint is deleted at the end.
    """
    cells, gids, values = [], [], []
    n_done = 0
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as ck:
            done = ck["done"]
            # A checkpoint without token_ig holds only |IG| (from before the signed IG was kept)
            if "token_ig" in ck.files and done.size <= order.size and np.array_equal(done, order[:done.size]):
                n_done = int(done.size)
                cells, gids, values = [ck["token_cells"]], [ck["token_gids"]], [ck["token_ig"]]
                print(f"  IG checkpoint: resuming after {n_done:,} of {order.size:,} cells", flush=True)
            else:
                print(f"  IG checkpoint {checkpoint.name} is for other cells, starting over", flush=True)

    def save() -> None:
        tmp = checkpoint.with_name(checkpoint.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, done=order[:n_done], token_cells=np.concatenate(cells),
                     token_gids=np.concatenate(gids), token_ig=np.concatenate(values))
        os.replace(tmp, checkpoint)

    last_save = time.monotonic()
    with progress(total=order.size, initial=n_done, desc=desc, unit="cell") as bar:
        for idx in make_batches(order[n_done:]):
            batch_gids, batch_ig, keep = ig_fn(idx)
            cells.append(np.repeat(idx, keep.sum(axis=1)).astype(np.int32))
            gids.append(batch_gids[keep].astype(np.int32))
            values.append(batch_ig[keep].astype(np.float32))
            n_done += len(idx)
            bar.update(len(idx))
            if checkpoint is not None and time.monotonic() - last_save > CHECKPOINT_MINUTES * 60:
                save()
                last_save = time.monotonic()

    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    return cat(cells, np.int32), cat(gids, np.int32), cat(values, np.float32)


def remove_checkpoint(checkpoint: Path) -> None:
    """Delete the IG checkpoint once the cell type's outputs are written."""
    checkpoint.unlink(missing_ok=True)


def _named_means(gids: np.ndarray, weights: np.ndarray, n_cells: int, names: dict) -> Dict[str, float]:
    """Mean of `weights` per token id over the cells that have it, named by `names` (token id -> gene).

    `gids` are the tokens of `n_cells` cells. A cell has each gene once, so a
    gene's token count is its number of cells, and only genes in at least
    C.MIN_PRESENT_FRAC of the `n_cells` get a mean.
    """
    if gids.size == 0 or n_cells == 0:
        return {}
    sums = np.bincount(gids, weights=weights)
    cells_with_gene = np.bincount(gids)
    seen_ids = np.unique(gids)
    return {names[int(gid)]: float(sums[gid]) / cells_with_gene[gid] for gid in seen_ids
            if cells_with_gene[gid] >= C.MIN_PRESENT_FRAC * n_cells}


@dataclass
class IGResult:
    """Pooled IG scores of one cell type, plus what the stratified (CSSI) and signed scores need."""
    mean_abs: Dict[str, float]  # mean |IG| per gene over the scored cells with it
    gate: dict
    embeddings: np.ndarray      # (n_cells, d) residual-stream CLS embedding entering the attention layer
    labels: np.ndarray          # (n_cells,) 0 = control, 1 = PD
    correct: np.ndarray         # (n_cells,) True on the scored cells: the correctly classified ones, or all
    token_cells: np.ndarray     # (n_tokens,) cell of every gene token of the scored cells, in cell order
    token_gids: np.ndarray      # (n_tokens,) its token id
    token_ig: np.ndarray        # (n_tokens,) its signed IG (|IG| only in files saved before it was kept)

    @property
    def token_abs_ig(self) -> np.ndarray:
        return np.abs(self.token_ig)

    @classmethod
    def from_tokens(cls, token_cells, token_gids, token_ig, embeddings, labels, correct,
                    names: dict) -> "IGResult":
        # The mean |IG| per gene over the scored cells that have it, for the
        # genes in at least C.MIN_PRESENT_FRAC of them
        n_used = int(correct.sum())
        ok = n_used > 0
        gate = {"ok": ok, "reason": "" if ok else "no correctly-classified cells", "mode": "ig_predictions"}
        return cls(mean_abs=_named_means(token_gids, np.abs(token_ig), n_used, names), gate=gate,
                   embeddings=embeddings, labels=labels, correct=correct, token_cells=token_cells,
                   token_gids=token_gids, token_ig=token_ig)

    def save(self, path) -> None:
        """Per-cell token IG (signed) to one .npz (embeddings excluded: cell_embeddings.npy)."""
        np.savez(path, labels=self.labels, correct=self.correct, token_cells=self.token_cells,
                 token_gids=self.token_gids, token_ig=self.token_ig)

    @classmethod
    def load(cls, path, embeddings: np.ndarray, names: dict) -> "IGResult":
        """A saved IGResult; the pooled mean |IG| is recomputed (equal up to summation order).

        Files saved before the signed IG was kept hold `token_abs_ig`, read as
        token_ig: their `IG` scores are right, their signed ones are not.
        """
        with np.load(path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        if "token_abs_ig" in data:
            data["token_ig"] = data.pop("token_abs_ig")
        return cls.from_tokens(embeddings=embeddings, names=names, **data)

    def to_signed_cache(self, names: dict, embedding_source: str) -> AttentionCache:
        """The signed IG as an AttentionCache (score IG±): its PD - control mean, CSSI and selections.

        Every cell is in the cache (cells without IG have no tokens); the
        scored cells are `correct`, passed as the cell mask of the attention
        outputs. Genes are named by `names` (token id -> current symbol).
        """
        n_cells = int(self.labels.size)
        if self.token_cells.size and np.any(np.diff(self.token_cells) < 0):
            raise ValueError("IG tokens are not in cell order")
        lens = np.bincount(self.token_cells, minlength=n_cells)
        uniq, inv = np.unique(self.token_gids, return_inverse=True)
        return AttentionCache(
            gene_idx=inv.astype(np.int32), attn=self.token_ig.astype(np.float32),
            cell_ptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            labels=self.labels, embeddings=self.embeddings,
            genes=np.asarray([str(names[int(g)]) for g in uniq]),
            embedding_source=embedding_source, correct=self.correct)


def stratified_ig_scores(result: IGResult, strata: np.ndarray, names: dict,
                         min_cells_per_condition: int = 20) -> Tuple[Dict[str, float], int]:
    """CSSI-max IG: per gene, the maximum over strata of its mean |IG| in the stratum.

    The mean is over the stratum's scored (e.g. correctly classified) cells
    that have the gene (at least C.MIN_PRESENT_FRAC of them), and a stratum
    needs min_cells_per_condition of them
    per condition, as the attention CSSI (the LoRA embedding groups cells by
    the model's prediction, so few strata qualify). Returns (scores, number
    of valid strata); without any valid stratum the scores are the pooled ones.
    """
    token_strata = strata[result.token_cells]
    best: Dict[str, float] = {}
    valid = 0
    for stratum in np.unique(strata):
        cells = (strata == stratum) & result.correct
        counts = [int(np.sum(cells & (result.labels == c))) for c in (0, 1)]
        if min(counts) < min_cells_per_condition:
            continue
        valid += 1
        tokens = token_strata == stratum
        means = _named_means(result.token_gids[tokens], np.abs(result.token_ig[tokens]), int(cells.sum()), names)
        for gene, score in means.items():
            if gene not in best or score > best[gene]:
                best[gene] = score
    if valid == 0:
        return dict(result.mean_abs), 0
    return best, valid


# ---------------------------------------------------------------------------
# Integrated Gradients of one output w.r.t. the input expression (scGPT)
# ---------------------------------------------------------------------------
def _pd_logit(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Single PD logit per cell (n_cls=1 sigmoid head)."""
    out = model(gids, vals, src_key_padding_mask=mask, batch_labels=None,
                CLS=True, CCE=False, MVC=False, ECS=False)
    return out["cls_output"].squeeze(1)


def _cell_embedding(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """scGPT's cell embedding (the <cls> output of the last layer), no head."""
    out = model(gids, vals, src_key_padding_mask=mask, batch_labels=None,
                CLS=False, CCE=False, MVC=False, ECS=False)
    return out["cell_emb"]


def _signature_output(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor,
                      mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """SIGnature's single output per cell: sum_k |w_k * f_k(x')|, w = the real cell's embedding."""
    return torch.sum(torch.abs(weights * _cell_embedding(model, gids, vals, mask)), dim=1)


# Integrated Gradients of the PD logit or of the SIGnature output w.r.t. the
# input expression, one row per cell. Baseline is a zero vector
def _ig_batch(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor,
              mask: torch.Tensor, device: torch.device, n_steps: int,
              internal_batch_size: list, output: str = "pd_logit") -> np.ndarray:
    amp = device.type == "cuda"

    # Function for forward the model
    def forward_fn(v: torch.Tensor, gids_: torch.Tensor, mask_: torch.Tensor, *weights) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=amp):
            if output == "signature":
                return _signature_output(model, gids_, v, mask_, weights[0]).float()
            return _pd_logit(model, gids_, v, mask_)

    extra = (gids, mask)
    if output == "signature":
        # SIGnature's weights: the embedding of the real cell, fixed along the path
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            extra = (gids, mask, _cell_embedding(model, gids, vals, mask).float())

    # Integrated Gradients attribution for the batch of cells. Without
    # internal_batch_size Captum runs every step of every cell in one forward
    # (8 cells x 50 steps x 1201 tokens), which does not fit in the GPU
    attributor = IntegratedGradients(forward_fn, multiply_by_inputs=True)
    attrs = attribute(
        attributor, internal_batch_size,
        inputs=vals,
        baselines=torch.zeros_like(vals),
        additional_forward_args=extra,
        n_steps=n_steps,
        method="gausslegendre",
    )
    return attrs.detach().float().cpu().numpy()


def _signature_normalise(ig: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Each cell's IG scaled so the |IG| of its gene tokens sum to SIGNATURE_TARGET_SUM (sign kept)."""
    sums = np.where(keep, np.abs(ig), 0.0).sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return ig * (SIGNATURE_TARGET_SUM / sums)


def compute_ig_result(
    model: torch.nn.Module, pt: Dict[str, torch.Tensor], cache, vocab, gene_names: dict,
    device: torch.device, n_steps: int = IG_STEPS, batch_size: int = IG_BATCH_SIZE,
    internal_batch_size: int = IG_INTERNAL_BATCH_SIZE, checkpoint: Optional[Path] = None,
    desc: str = "IG", output: str = "pd_logit",
) -> IGResult:
    """Signed IG of every gene token of the scored cells, and their mean |IG| per gene.

    `cache` is the model's AttentionCache of the same cells (collect_scgpt):
    its embeddings are the result's. With output="pd_logit" (classifiers,
    collect_scgpt with use_predictions) the scored cells are the correctly
    classified ones; with output="signature" (encoders without a head) every
    cell, each normalised as SIGnature does.
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}")
    pad_id = vocab[C.PAD_TOKEN]
    special_arr = np.array(sorted({vocab[t] for t in C.SPECIAL_TOKENS if t in vocab} | {pad_id}))
    scored = (np.ones(cache.n_cells, dtype=bool) if output == "signature"
              else np.asarray(cache.correct, dtype=bool))

    # We only need gradients w.r.t. the input values, not the (frozen) weights.
    for p in model.parameters():
        p.requires_grad_(False)

    internal = [internal_batch_size]

    # IG of a batch of cells and its gene tokens (position 0 is the CLS)
    def ig_fn(idx: np.ndarray):
        gids = pt["gene_ids"][idx].to(device)
        vals = pt["values"][idx].to(device).float()
        ig = _ig_batch(model, gids, vals, gids.eq(pad_id), device, n_steps, internal, output)
        gids_np = gids.cpu().numpy()
        keep = ~np.isin(gids_np, special_arr)
        keep[:, 0] = False
        if output == "signature":
            ig = _signature_normalise(ig, keep)
        return gids_np, ig, keep

    make_batches = lambda order: [order[i:i + batch_size] for i in range(0, order.size, batch_size)]
    tokens = run_ig(np.where(scored)[0], make_batches, ig_fn, checkpoint, desc)
    return IGResult.from_tokens(*tokens, embeddings=cache.embeddings, labels=cache.labels,
                                correct=scored, names=gene_names)
