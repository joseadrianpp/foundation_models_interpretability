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

As for scGPT, the correctly classified cells and the cell embeddings come from
the LoRA model's attention pass (attention_cssi_leiden.collect_geneformer, the
one of the attention_lora_<model> task), and the IG computed so far is saved
every few minutes so a rerun resumes it (step7_explainability_ig_scgpt.run_ig).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from captum.attr import IntegratedGradients

from ._step7_geneformer import _pad_batch
from .step7_explainability_ig_scgpt import IG_STEPS, IGResult, attribute, run_ig

# Cells per batch are set by a token budget: the cells are attributed longest
# first, and a batch holds as many cells as fit IG_TOKENS_PER_BATCH padded
# tokens (3 Geneformer cells of 4096 tokens on an 8 GB GPU; --batch-scale
# multiplies it). Each Captum forward runs one step of the whole batch, in bf16
IG_TOKENS_PER_BATCH = 3 * 4096


# Integrated Gradients of the PD logit w.r.t. the token embeddings, one row per cell
def _ig_batch(model: torch.nn.Module, ids: torch.Tensor, attn: torch.Tensor,
              mask_id: int, device: torch.device, n_steps: int, internal_batch_size: list) -> np.ndarray:

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
    # len(ids) sequences per step unless the GPU runs out of memory
    attributor = IntegratedGradients(forward_fn, multiply_by_inputs=True)
    attrs = attribute(
        attributor, internal_batch_size,
        inputs=real,
        baselines=baseline,
        additional_forward_args=(attn,),
        n_steps=n_steps,
        method="gausslegendre",
    )

    # One attribution per token: the sum over the embedding dimensions
    return attrs.sum(dim=-1).detach().float().cpu().numpy()


# Batches of cells already sorted longest first: each holds as many cells as fit
# the token budget at the length of its first (longest) cell
def _token_budget_batches(order: np.ndarray, lengths: List[int], tokens_per_batch: int) -> List[np.ndarray]:
    batches, start = [], 0
    while start < order.size:
        n = max(1, tokens_per_batch // lengths[order[start]])
        batches.append(order[start:start + n])
        start += n
    return batches


def compute_ig_result(
    model: torch.nn.Module, pt: Dict, cache, token_dict: dict, gene_names: dict,
    device: torch.device, n_steps: int = IG_STEPS, tokens_per_batch: int = IG_TOKENS_PER_BATCH,
    checkpoint: Optional[Path] = None, desc: str = "IG",
) -> IGResult:
    """Mean absolute IG per gene over the correctly classified cells, keeping every token's |IG|.

    `cache` is the LoRA model's AttentionCache of the same cells
    (collect_geneformer with use_predictions): its correctly classified cells
    are the ones attributed, its embeddings are the result's.
    """
    # Special tokens (<pad>, <mask>, <cls> and, in Geneformer, <eos>)
    pad_id, mask_id = token_dict["<pad>"], token_dict["<mask>"]
    special_arr = np.array(sorted(v for k, v in token_dict.items() if k.startswith("<")))

    # We only need gradients w.r.t. the input embeddings, not the (frozen) weights.
    for p in model.parameters():
        p.requires_grad_(False)

    # Padded tokens per Captum forward: one step of a whole batch, less if the
    # GPU ran out of memory on an earlier batch (kept as tokens, since the
    # batches get more cells as the cells get shorter)
    forward_tokens = [tokens_per_batch]

    def ig_fn(idx: np.ndarray):
        ids, attn = _pad_batch([pt["input_ids"][i] for i in idx], pad_id, device)
        internal = [max(1, min(len(idx), forward_tokens[0] // ids.shape[1]))]
        ig = _ig_batch(model, ids, attn, mask_id, device, n_steps, internal)
        if internal[0] < len(idx):
            forward_tokens[0] = min(forward_tokens[0], internal[0] * ids.shape[1])
        gids = ids.cpu().numpy()
        keep = ~np.isin(gids, special_arr)
        keep[:, 0] = False
        return gids, np.abs(ig), keep

    # Longest cells first, so each batch pads to cells of similar length and a
    # batch too large for the GPU fails at the start. The scores do not depend
    # on the order
    lengths = [len(s) for s in pt["input_ids"]]
    correct_idx = np.where(cache.correct)[0]
    correct_idx = correct_idx[np.argsort([-lengths[i] for i in correct_idx], kind="stable")]
    make_batches = lambda order: _token_budget_batches(order, lengths, tokens_per_batch)
    tokens = run_ig(correct_idx, make_batches, ig_fn, checkpoint, desc)
    return IGResult.from_tokens(*tokens, embeddings=cache.embeddings, labels=cache.labels,
                                correct=cache.correct, names=gene_names)
