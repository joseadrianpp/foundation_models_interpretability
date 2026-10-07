"""
Integrated Gradients (IG) attribution for the LoRA fine-tuned Geneformer and
scCello models, the same score as the scGPT one (step7_explainability_ig_scgpt.py).

Geneformer and scCello have no expression value to integrate over: a cell is
the list of its genes ranked by expression, so the expression level is the
position and the input is a token. IG therefore runs over the token embeddings,
as in pipeline_unified:
  - The baseline replaces every gene embedding by the <mask> embedding, the
    only special token the model was trained to read as an input ("a gene is
    here, identity unknown"). The <cls> token and the positions are the same
    in both endpoints, so only the identity of each gene is attributed.
  - IG(gene) = sum over the embedding dimensions of
    (real - baseline) * integrated_gradient, its signed total contribution.
  - We keep only correctly classified cells and score each gene by the MEAN
    ABSOLUTE attribution over all of them, a cell without the gene counting
    as 0 (non-negative, like RF importance).
"""

from typing import Dict, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from ._step7_geneformer import ATTN_BATCH_SIZE, _add_cell, _pad_batch
from .step7_explainability_ig_scgpt import IG_STEPS

# Geneformer cells have up to 4096 tokens and IG needs a backward pass per step,
# so the batch is the one pipeline_unified uses (it fits in 24 GB with bf16)
IG_BATCH_SIZE = 10


# Integrated Gradients of the PD logit w.r.t. the token embeddings, one row per cell
def _ig_batch(model: torch.nn.Module, ids: torch.Tensor, attn: torch.Tensor,
              mask_id: int, device: torch.device, n_steps: int) -> np.ndarray:

    # Real endpoint: the embedding of every token of the cell
    embed = model.backbone.bert.embeddings.word_embeddings
    with torch.no_grad():
        real = embed(ids)

    # Baseline: every gene becomes <mask>, the CLS keeps its own embedding
    baseline = embed.weight[mask_id].detach().expand_as(real).clone()
    baseline[:, 0, :] = real[:, 0, :]

    # Function for forward the model. bf16 only inside the attributed forward,
    # which is what fits the 4096-token cells in memory; Captum integrates in fp32
    def forward_fn(e: torch.Tensor, attn_: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            return model(inputs_embeds=e, attention_mask=attn_).float()

    # Integrated Gradients attribution for the batch of cells, one forward of
    # len(ids) sequences per step (internal_batch_size)
    attributor = IntegratedGradients(forward_fn, multiply_by_inputs=True)
    attrs = attributor.attribute(
        inputs=real,
        baselines=baseline,
        additional_forward_args=(attn,),
        n_steps=n_steps,
        method="gausslegendre",
        internal_batch_size=len(ids),
    )

    # One attribution per token: the sum over the embedding dimensions
    return attrs.sum(dim=-1).detach().float().cpu().numpy()


# Mean absolute integrated gradient per gene over correctly classified cells
def _compute_ig_importance(
    model: torch.nn.Module, pt: Dict, token_dict: dict, gene_names: dict,
    device: torch.device, n_steps: int = IG_STEPS, batch_size: int = IG_BATCH_SIZE,
) -> Tuple[Dict[str, float], dict]:

    # Number of cells and special tokens (<pad>, <mask>, <cls> and, in Geneformer, <eos>)
    n_cells = len(pt["input_ids"])
    pad_id, mask_id = token_dict["<pad>"], token_dict["<mask>"]
    special_ids = {v for k, v in token_dict.items() if k.startswith("<")}

    # We only need gradients w.r.t. the input embeddings, not the (frozen) weights.
    for p in model.parameters():
        p.requires_grad_(False)

    # Pass 1: classify every cell (no gradients needed, just inference), in fp32
    # so the correctly classified cells do not depend on the bf16 of pass 2
    labels_all = pt["condition_labels"].numpy()
    preds_all = np.empty(n_cells, dtype=np.int64)
    for start in range(0, n_cells, ATTN_BATCH_SIZE):
        end = min(start + ATTN_BATCH_SIZE, n_cells)
        ids, attn = _pad_batch(pt["input_ids"][start:end], pad_id, device)
        with torch.no_grad():
            preds_all[start:end] = (model(input_ids=ids, attention_mask=attn) > 0).long().cpu().numpy()

    # Filter the correctly classified cells
    correct_idx = np.where(preds_all == labels_all)[0]
    n_used = len(correct_idx)

    # Longest cells first, so each batch pads to cells of similar length and a
    # batch too large for the GPU fails at the start. The sum below does not
    # depend on the order
    lengths = [-len(pt["input_ids"][i]) for i in correct_idx]
    correct_idx = correct_idx[np.argsort(lengths, kind="stable")]

    # Pass 2: Integrated Gradients, only on the correctly classified subset.
    sum_abs: Dict[int, float] = {}
    for start in range(0, n_used, batch_size):
        idx = correct_idx[start:start + batch_size]
        ids, attn = _pad_batch([pt["input_ids"][i] for i in idx], pad_id, device)

        # Compute IG for the batch and add its absolute value gene by gene
        ig = _ig_batch(model, ids, attn, mask_id, device, n_steps)
        gids = ids.cpu().numpy()
        for i in range(len(idx)):
            _add_cell(sum_abs, gids[i], np.abs(ig[i]), special_ids)

    # Compute the mean IG per gene over every correctly classified cell, a cell
    # without the gene adds 0, so a gene seen in a handful of cells cannot
    # outrank one seen in all of them
    mean_abs: Dict[str, float] = {}
    for gid, s in sum_abs.items():
        mean_abs[gene_names[gid]] = s / n_used

    # Safe case for 0% accuracy
    ok = n_used > 0
    gate = {"ok": bool(ok),
            "reason": "" if ok else "no correctly-classified cells",
            "mode": "ig_predictions"}
    return mean_abs, gate
