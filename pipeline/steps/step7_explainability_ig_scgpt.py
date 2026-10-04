"""
Integrated Gradients (IG) attribution for the LoRA fine-tuned scGPT model.

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
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from . import _step7_common as common
from .. import config as C
from ..utils.progress import progress

# Gauss-Legendre interpolation points and forward batch size. IG needs a
# backward pass per step, so the batch is smaller than the attention path's.
IG_STEPS = 24
IG_BATCH_SIZE = 8
# Captum runs IG_BATCH_SIZE * IG_STEPS interpolated cells per forward/backward;
# splitting them into chunks of this size keeps an 8 GB GPU from running out of
# memory and does not change the attributions (the gradients are just summed)
IG_INTERNAL_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Integrated Gradients of the PD logit w.r.t. the input expression
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
              internal_batch_size: int = IG_INTERNAL_BATCH_SIZE) -> np.ndarray:

    # Function for forward the model
    def forward_fn(v: torch.Tensor, gids_: torch.Tensor, mask_: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            return _pd_logit(model, gids_, v, mask_)

    # Integrated Gradients attribution for the batch of cells
    attributor = IntegratedGradients(forward_fn, multiply_by_inputs=True)
    attrs = attributor.attribute(
        inputs=vals,
        baselines=torch.zeros_like(vals),
        additional_forward_args=(gids, mask),
        n_steps=n_steps,
        method="gausslegendre",
        internal_batch_size=internal_batch_size,
    )
    return attrs.detach().float().cpu().numpy()



@dataclass
class IGResult:
    """Pooled IG scores of one cell type, plus what the stratified (CSSI) score needs."""
    mean_abs: Dict[str, float]  # mean |IG| per gene over the correctly classified cells
    gate: dict
    embeddings: np.ndarray      # (n_cells, d) residual-stream CLS embedding entering layer NUM_ATTN_LAYERS
    labels: np.ndarray          # (n_cells,) 0 = control, 1 = PD
    correct: np.ndarray         # (n_cells,) True where the classifier predicts the true label
    token_cells: np.ndarray     # (n_tokens,) cell of every gene token of the correctly classified cells
    token_gids: np.ndarray      # (n_tokens,) its token id
    token_abs_ig: np.ndarray    # (n_tokens,) its |IG|

    def save(self, path) -> None:
        """Per-cell token |IG| to one .npz (embeddings excluded: cell_embeddings.npy)."""
        np.savez(path, labels=self.labels, correct=self.correct, token_cells=self.token_cells,
                 token_gids=self.token_gids, token_abs_ig=self.token_abs_ig)

    @classmethod
    def load(cls, path, embeddings: np.ndarray, vocab) -> "IGResult":
        """A saved IGResult; the pooled mean |IG| is recomputed (equal up to summation order)."""
        with np.load(path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        n_used = int(data["correct"].sum())
        sums = np.bincount(data["token_gids"], weights=data["token_abs_ig"], minlength=len(vocab))
        seen_ids = np.unique(data["token_gids"])
        mean_abs = {name: float(sums[gid]) / n_used
                    for gid, name in zip(seen_ids, vocab.lookup_tokens(seen_ids.tolist()))
                    if name not in C.SPECIAL_TOKENS}
        return cls(mean_abs=mean_abs, gate={"ok": n_used > 0, "reason": "", "mode": "ig_predictions"},
                   embeddings=embeddings, **data)


# Mean absolute integrated gradient per gene over correctly classified cells
def _compute_ig_importance(
    model: torch.nn.Module, pt: Dict[str, torch.Tensor], vocab,
    device: torch.device, n_steps: int = IG_STEPS, batch_size: int = IG_BATCH_SIZE,
    classify_batch_size: int = 64, internal_batch_size: int = IG_INTERNAL_BATCH_SIZE,
    desc: str = "IG",
) -> Tuple[Dict[str, float], dict]:
    result = _compute_ig_result(model, pt, vocab, device, n_steps, batch_size, classify_batch_size,
                                internal_batch_size, desc)
    return result.mean_abs, result.gate


# The same, keeping every cell's embedding and every token's |IG| (IGResult)
def _compute_ig_result(
    model: torch.nn.Module, pt: Dict[str, torch.Tensor], vocab,
    device: torch.device, n_steps: int = IG_STEPS, batch_size: int = IG_BATCH_SIZE,
    classify_batch_size: int = 64, internal_batch_size: int = IG_INTERNAL_BATCH_SIZE,
    desc: str = "IG",
) -> IGResult:

    # Number of cells and special tokens
    n_cells = pt["gene_ids"].shape[0]
    pad_id = vocab[C.PAD_TOKEN]
    special_ids = {vocab[t] for t in C.SPECIAL_TOKENS if t in vocab}
    special_arr = np.array(sorted(special_ids | {pad_id}))

    # We only need gradients w.r.t. the input values, not the (frozen) weights.
    for p in model.parameters():
        p.requires_grad_(False)

    # Pass 1: classify every cell (no gradients needed, just inference).
    # Initialize variables
    labels_all = pt["condition_labels"].numpy()
    preds_all = np.empty(n_cells, dtype=np.int64)
    # The same pass keeps each cell's CLS embedding entering the attention
    # layer (as the zero-shot attention task does), for the CSSI strata
    embeddings = []
    capture = common._LayerInputCapture(model.transformer_encoder.layers[common.NUM_ATTN_LAYERS])
    try:
        # Iterate over cells and classified them
        for start in progress(range(0, n_cells, classify_batch_size), desc=f"{desc} classify", unit="batch"):
            # Special case for last batch
            end = min(start + classify_batch_size, n_cells)
            gids = pt["gene_ids"][start:end].to(device)
            vals = pt["values"][start:end].to(device).float()
            mask = gids.eq(pad_id)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds_all[start:end] = (_pd_logit(model, gids, vals, mask) > 0).long().cpu().numpy()
            embeddings.append(capture.captured[:, 0, :].float().cpu().numpy())
    finally:
        capture.remove()

    # Filter the correctly classified cells
    correct_mask = preds_all == labels_all
    correct_idx = torch.from_numpy(np.where(correct_mask)[0])
    n_used = len(correct_idx)

    # Pass 2: Integrated Gradients, only on the correctly classified subset.
    # Initialize variables: per token id, summed |IG| and whether it was seen
    n_ids = len(vocab)
    sum_abs = np.zeros(n_ids)
    seen = np.zeros(n_ids, dtype=bool)
    token_cells, token_gids, token_abs_ig = [], [], []

    # Iterate over the correctly classified cells in batches, computing IG for each
    bar = progress(total=n_used, desc=desc, unit="cell")
    for start in range(0, n_used, batch_size):
        # Special case for last batch
        end = min(start + batch_size, n_used)

        # Correct id cells
        idx = correct_idx[start:end]

        # Genes to tensors for gene ids and labels (which we don't currently have)
        gids = pt["gene_ids"][idx].to(device)
        vals = pt["values"][idx].to(device).float()

        # Masking the padding values
        mask = gids.eq(pad_id)

        # Compute IG for the batch of correctly classified cells
        ig = _ig_batch(model, gids, vals, mask, device, n_steps, internal_batch_size)
        gids_np = gids.cpu().numpy()

        # We have the IG by cell, like in attention, but we want the
        # mean absolute IG per gene over only correctly classified cells.
        # Every gene token of the batch at once (position 0 is the CLS)
        keep = ~np.isin(gids_np, special_arr)
        keep[:, 0] = False
        sum_abs += np.bincount(gids_np[keep], weights=np.abs(ig[keep]), minlength=n_ids)
        seen[gids_np[keep]] = True
        token_cells.append(np.repeat(idx.numpy(), keep.sum(axis=1)).astype(np.int32))
        token_gids.append(gids_np[keep].astype(np.int32))
        token_abs_ig.append(np.abs(ig[keep]).astype(np.float32))
        bar.update(end - start)
    bar.close()

    # Compute the mean IG per gene over every correctly classified cell, a cell
    # without the gene adds 0, so a gene seen in a handful of cells cannot
    # outrank one seen in all of them
    mean_abs: Dict[str, float] = {}
    seen_ids = np.where(seen)[0]
    for gid, name in zip(seen_ids, vocab.lookup_tokens(seen_ids.tolist())):
        if name in C.SPECIAL_TOKENS:
            continue
        mean_abs[name] = float(sum_abs[gid]) / n_used

    # Safe case for 100% or 0% accuracy
    ok = n_used > 0
    gate = {"ok": bool(ok),
            "reason": "" if ok else "no correctly-classified cells",
            "mode": "ig_predictions"}
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    return IGResult(mean_abs=mean_abs, gate=gate, embeddings=np.concatenate(embeddings), labels=labels_all,
                    correct=correct_mask, token_cells=cat(token_cells, np.int32),
                    token_gids=cat(token_gids, np.int32), token_abs_ig=cat(token_abs_ig, np.float32))


def stratified_ig_scores(result: IGResult, strata: np.ndarray, vocab,
                         min_cells_per_condition: int = 20) -> Tuple[Dict[str, float], int]:
    """CSSI-max IG: per gene, the maximum over strata of its mean |IG| in the stratum.

    The mean is over the stratum's correctly classified cells (a cell without
    the gene adding 0), and a stratum needs min_cells_per_condition of them
    per condition, as the attention CSSI (the LoRA embedding groups cells by
    the model's prediction, so few strata qualify). Returns (scores, number
    of valid strata); without any valid stratum the scores are the pooled ones.
    """
    n_ids = len(vocab)
    best = np.zeros(n_ids)
    seen = np.zeros(n_ids, dtype=bool)
    token_strata = strata[result.token_cells]
    valid = 0
    for stratum in np.unique(strata):
        cells = (strata == stratum) & result.correct
        counts = [int(np.sum(cells & (result.labels == c))) for c in (0, 1)]
        if min(counts) < min_cells_per_condition:
            continue
        valid += 1
        tokens = token_strata == stratum
        gids = result.token_gids[tokens]
        means = np.bincount(gids, weights=result.token_abs_ig[tokens], minlength=n_ids) / cells.sum()
        present = np.bincount(gids, minlength=n_ids) > 0
        best[present] = np.where(seen[present], np.maximum(best[present], means[present]), means[present])
        seen |= present
    if valid == 0:
        return dict(result.mean_abs), 0
    seen_ids = np.where(seen)[0]
    return {name: float(best[gid]) for gid, name in zip(seen_ids, vocab.lookup_tokens(seen_ids.tolist()))
            if name not in C.SPECIAL_TOKENS}, valid
