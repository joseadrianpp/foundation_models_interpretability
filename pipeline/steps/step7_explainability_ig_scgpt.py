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
"""

from typing import Dict, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from .. import config as C

# Gauss-Legendre interpolation points and forward batch size. IG needs a
# backward pass per step, so the batch is smaller than the attention path's.
IG_STEPS = 24
IG_BATCH_SIZE = 8

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
              mask: torch.Tensor, device: torch.device, n_steps: int) -> np.ndarray:

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
    )
    return attrs.detach().float().cpu().numpy()



# Mean absolute integrated gradient per gene over correctly classified cells
def _compute_ig_importance(
    model: torch.nn.Module, pt: Dict[str, torch.Tensor], vocab,
    device: torch.device, n_steps: int = IG_STEPS, batch_size: int = IG_BATCH_SIZE,
    classify_batch_size: int = 64,
) -> Tuple[Dict[str, float], dict]:

    # Number of cells and special tokens
    n_cells = pt["gene_ids"].shape[0]
    pad_id = vocab[C.PAD_TOKEN]
    special_ids = {vocab[t] for t in C.SPECIAL_TOKENS if t in vocab}

    # We only need gradients w.r.t. the input values, not the (frozen) weights.
    for p in model.parameters():
        p.requires_grad_(False)

    # Pass 1: classify every cell (no gradients needed, just inference).
    # Initialize variables
    labels_all = pt["condition_labels"].numpy()
    preds_all = np.empty(n_cells, dtype=np.int64)
    # Iterate over cells and classified them
    for start in range(0, n_cells, classify_batch_size):
        # Special case for last batch
        end = min(start + classify_batch_size, n_cells)
        gids = pt["gene_ids"][start:end].to(device)
        vals = pt["values"][start:end].to(device).float()
        mask = gids.eq(pad_id)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            preds_all[start:end] = (_pd_logit(model, gids, vals, mask) > 0).long().cpu().numpy()

    # Filter the correctly classified cells
    correct_mask = preds_all == labels_all
    correct_idx = torch.from_numpy(np.where(correct_mask)[0])
    n_used = len(correct_idx)

    # Pass 2: Integrated Gradients, only on the correctly classified subset.
    # Initialize variables
    sum_abs: Dict[int, float] = {}

    # Iterate over the correctly classified cells in batches, computing IG for each
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
        ig = _ig_batch(model, gids, vals, mask, device, n_steps)
        gids_np = gids.cpu().numpy()

        # We have the IG by cell, like in attention, but we want the 
        # mean absolute IG per gene over only correctly classified cells
        for i in range(end - start):
            for pos in range(1, gids_np.shape[1]):
                gid = int(gids_np[i, pos])
                if gid == pad_id or gid in special_ids:
                    continue
                sum_abs[gid] = sum_abs.get(gid, 0.0) + abs(float(ig[i, pos]))

    # Compute the mean IG per gene over every correctly classified cell, a cell
    # without the gene adds 0, so a gene seen in a handful of cells cannot
    # outrank one seen in all of them
    mean_abs: Dict[str, float] = {}
    for gid, s in sum_abs.items():
        name = vocab.lookup_tokens([gid])[0]
        if name in C.SPECIAL_TOKENS:
            continue
        mean_abs[name] = s / n_used

    # Safe case for 100% or 0% accuracy
    ok = n_used > 0
    gate = {"ok": bool(ok),
            "reason": "" if ok else "no correctly-classified cells",
            "mode": "ig_predictions"}
    return mean_abs, gate
