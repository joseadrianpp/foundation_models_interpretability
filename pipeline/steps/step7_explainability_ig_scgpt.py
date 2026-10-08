"""
Integrated Gradients (IG) attribution for the LoRA fine-tuned scGPT model,
plus what every model's IG shares (IGResult, the checkpointed IG loop and the
CSSI version), used by step7_explainability_ig_geneformer.py too.

We use Captum's IntegratedGradients (Gauss-Legendre quadrature,
n_steps interpolation points)

Why IG and not attention: attention tells us what the model looks at, not
what drives its decision. IG measures, for every gene, how much that gene's
measured expression pushes the cell towards the PD prediction.

How it works:
  - Captum integrates the gradient of the PD logit along the straight path
    from baseline to the real cell, using `n_steps` Gauss-Legendre points.
  - IG(gene) = (real_value - baseline_value) * integrated_gradient.
  - We keep only correctly classified cells and score each gene by the MEAN
    ABSOLUTE attribution over all of them, a cell without the gene counting
    as 0 (non-negative, like RF importance).
  - The CSSI version (stratified_ig_scores) takes, per gene, the maximum of
    that mean over the Leiden strata of the LoRA model's own cell embedding.

The correctly classified cells and the cell embeddings come from the LoRA
model's attention pass (attention_cssi_leiden.collect_scgpt), the same one the
attention_lora_scgpt task runs, so the two tasks share their Leiden strata.
IG over every correctly classified cell takes hours on the large cell types,
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

    ig_fn(cell indices) returns the batch's token ids (B, L), |IG| (B, L) and
    a (B, L) mask of the gene tokens to keep. Returns the cell, token id and
    |IG| of every kept token, cell after cell in `order`. With `checkpoint`,
    the tokens of the cells done so far go to that .npz every
    CHECKPOINT_MINUTES; a rerun whose `order` starts with those cells resumes
    after them (the batches may differ, e.g. on another GPU: each cell's IG
    does not depend on the others). The checkpoint is deleted at the end.
    """
    cells, gids, abs_ig = [], [], []
    n_done = 0
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as ck:
            done = ck["done"]
            if done.size <= order.size and np.array_equal(done, order[:done.size]):
                n_done = int(done.size)
                cells, gids, abs_ig = [ck["token_cells"]], [ck["token_gids"]], [ck["token_abs_ig"]]
                print(f"  IG checkpoint: resuming after {n_done:,} of {order.size:,} cells", flush=True)
            else:
                print(f"  IG checkpoint {checkpoint.name} is for other cells, starting over", flush=True)

    def save() -> None:
        tmp = checkpoint.with_name(checkpoint.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, done=order[:n_done], token_cells=np.concatenate(cells),
                     token_gids=np.concatenate(gids), token_abs_ig=np.concatenate(abs_ig))
        os.replace(tmp, checkpoint)

    last_save = time.monotonic()
    with progress(total=order.size, initial=n_done, desc=desc, unit="cell") as bar:
        for idx in make_batches(order[n_done:]):
            batch_gids, batch_abs, keep = ig_fn(idx)
            cells.append(np.repeat(idx, keep.sum(axis=1)).astype(np.int32))
            gids.append(batch_gids[keep].astype(np.int32))
            abs_ig.append(batch_abs[keep].astype(np.float32))
            n_done += len(idx)
            bar.update(len(idx))
            if checkpoint is not None and time.monotonic() - last_save > CHECKPOINT_MINUTES * 60:
                save()
                last_save = time.monotonic()

    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    return cat(cells, np.int32), cat(gids, np.int32), cat(abs_ig, np.float32)


def remove_checkpoint(checkpoint: Path) -> None:
    """Delete the IG checkpoint once the cell type's outputs are written."""
    checkpoint.unlink(missing_ok=True)


def _named_means(gids: np.ndarray, weights: np.ndarray, n_cells: int, names: dict) -> Dict[str, float]:
    """Sum of `weights` per token id over `n_cells` cells, named by `names` (token id -> gene)."""
    if gids.size == 0 or n_cells == 0:
        return {}
    sums = np.bincount(gids, weights=weights)
    seen_ids = np.unique(gids)
    return {names[int(gid)]: float(sums[gid]) / n_cells for gid in seen_ids}


@dataclass
class IGResult:
    """Pooled IG scores of one cell type, plus what the stratified (CSSI) score needs."""
    mean_abs: Dict[str, float]  # mean |IG| per gene over the correctly classified cells
    gate: dict
    embeddings: np.ndarray      # (n_cells, d) residual-stream CLS embedding entering the attention layer
    labels: np.ndarray          # (n_cells,) 0 = control, 1 = PD
    correct: np.ndarray         # (n_cells,) True where the classifier predicts the true label
    token_cells: np.ndarray     # (n_tokens,) cell of every gene token of the correctly classified cells
    token_gids: np.ndarray      # (n_tokens,) its token id
    token_abs_ig: np.ndarray    # (n_tokens,) its |IG|

    @classmethod
    def from_tokens(cls, token_cells, token_gids, token_abs_ig, embeddings, labels, correct,
                    names: dict) -> "IGResult":
        # The mean IG per gene over every correctly classified cell, a cell
        # without the gene adds 0, so a gene seen in a handful of cells cannot
        # outrank one seen in all of them
        n_used = int(correct.sum())
        ok = n_used > 0
        gate = {"ok": ok, "reason": "" if ok else "no correctly-classified cells", "mode": "ig_predictions"}
        return cls(mean_abs=_named_means(token_gids, token_abs_ig, n_used, names), gate=gate,
                   embeddings=embeddings, labels=labels, correct=correct, token_cells=token_cells,
                   token_gids=token_gids, token_abs_ig=token_abs_ig)

    def save(self, path) -> None:
        """Per-cell token |IG| to one .npz (embeddings excluded: cell_embeddings.npy)."""
        np.savez(path, labels=self.labels, correct=self.correct, token_cells=self.token_cells,
                 token_gids=self.token_gids, token_abs_ig=self.token_abs_ig)

    @classmethod
    def load(cls, path, embeddings: np.ndarray, names: dict) -> "IGResult":
        """A saved IGResult; the pooled mean |IG| is recomputed (equal up to summation order)."""
        with np.load(path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        return cls.from_tokens(embeddings=embeddings, names=names, **data)


def stratified_ig_scores(result: IGResult, strata: np.ndarray, names: dict,
                         min_cells_per_condition: int = 20) -> Tuple[Dict[str, float], int]:
    """CSSI-max IG: per gene, the maximum over strata of its mean |IG| in the stratum.

    The mean is over the stratum's correctly classified cells (a cell without
    the gene adding 0), and a stratum needs min_cells_per_condition of them
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
        means = _named_means(result.token_gids[tokens], result.token_abs_ig[tokens], int(cells.sum()), names)
        for gene, score in means.items():
            if gene not in best or score > best[gene]:
                best[gene] = score
    if valid == 0:
        return dict(result.mean_abs), 0
    return best, valid


# ---------------------------------------------------------------------------
# Integrated Gradients of the PD logit w.r.t. the input expression (scGPT)
# ---------------------------------------------------------------------------
def _pd_logit(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Single PD logit per cell (n_cls=1 sigmoid head)."""
    out = model(gids, vals, src_key_padding_mask=mask, batch_labels=None,
                CLS=True, CCE=False, MVC=False, ECS=False)
    return out["cls_output"].squeeze(1)

# Integrated Gradients of the PD logit w.r.t. the input expression, one row per cell
# Baseline is a zero vector
def _ig_batch(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor,
              mask: torch.Tensor, device: torch.device, n_steps: int,
              internal_batch_size: list) -> np.ndarray:

    # Function for forward the model
    def forward_fn(v: torch.Tensor, gids_: torch.Tensor, mask_: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            return _pd_logit(model, gids_, v, mask_)

    # Integrated Gradients attribution for the batch of cells
    attributor = IntegratedGradients(forward_fn, multiply_by_inputs=True)
    attrs = attribute(
        attributor, internal_batch_size,
        inputs=vals,
        baselines=torch.zeros_like(vals),
        additional_forward_args=(gids, mask),
        n_steps=n_steps,
        method="gausslegendre",
    )
    return attrs.detach().float().cpu().numpy()


def compute_ig_result(
    model: torch.nn.Module, pt: Dict[str, torch.Tensor], cache, vocab, gene_names: dict,
    device: torch.device, n_steps: int = IG_STEPS, batch_size: int = IG_BATCH_SIZE,
    internal_batch_size: int = IG_INTERNAL_BATCH_SIZE, checkpoint: Optional[Path] = None,
    desc: str = "IG",
) -> IGResult:
    """Mean absolute IG per gene over the correctly classified cells, keeping every token's |IG|.

    `cache` is the LoRA model's AttentionCache of the same cells
    (collect_scgpt with use_predictions): its correctly classified cells are
    the ones attributed, its embeddings are the result's.
    """
    pad_id = vocab[C.PAD_TOKEN]
    special_arr = np.array(sorted({vocab[t] for t in C.SPECIAL_TOKENS if t in vocab} | {pad_id}))

    # We only need gradients w.r.t. the input values, not the (frozen) weights.
    for p in model.parameters():
        p.requires_grad_(False)

    internal = [internal_batch_size]

    # IG of a batch of correctly classified cells and its gene tokens (position 0 is the CLS)
    def ig_fn(idx: np.ndarray):
        gids = pt["gene_ids"][idx].to(device)
        vals = pt["values"][idx].to(device).float()
        ig = _ig_batch(model, gids, vals, gids.eq(pad_id), device, n_steps, internal)
        gids_np = gids.cpu().numpy()
        keep = ~np.isin(gids_np, special_arr)
        keep[:, 0] = False
        return gids_np, np.abs(ig), keep

    correct_idx = np.where(cache.correct)[0]
    make_batches = lambda order: [order[i:i + batch_size] for i in range(0, order.size, batch_size)]
    tokens = run_ig(correct_idx, make_batches, ig_fn, checkpoint, desc)
    return IGResult.from_tokens(*tokens, embeddings=cache.embeddings, labels=cache.labels,
                                correct=cache.correct, names=gene_names)
