"""
Shared logic for the attention-based and the IG interpretability task.

Pipeline:
  1. Tokenise input cells with `include_zero_gene=False`
  2. Forward-pass through the model, capturing the input to encoder layer 11
     via a forward pre-hook.
  3. Compute CLS-row attention scores from Q.Kt (per head), rank-normalise
     across the gene axis (ignoring padding), average across heads.
  4. Aggregate per-gene CLS attention per condition (attention_cssi_leiden.py).
     For the LoRA model we restrict to correctly classified cells.
  5. Score each gene with a user-supplied `score_fn(mean_pd, mean_ctrl)`.
"""

from typing import Callable, Dict, Optional

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from torch import nn

from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch

from .. import config as C
from ..utils.scgpt_model import FlexMLP, build_base_model, load_pretrained_weights
from ..utils.splits import labels_to_int

NUM_ATTN_LAYERS = 11        # 0-indexed transformer encoder layer for attention

# With every gene, the dense binned matrix of a whole cell type does not fit in
# memory (Dataset B has 171 780 oligodendrocytes), so cells are binned in chunks
BIN_CHUNK = 10000


# ---------------------------------------------------------------------------
# Tokenisation (matches Step 6: include_zero_gene=False, no normalisation/log)
# ---------------------------------------------------------------------------
def _tokenise(adata: sc.AnnData, vocab, seed: int = C.BINNING_SEED) -> Dict[str, torch.Tensor]:
    # scGPT can only embed genes that have a token, so anything outside the
    # vocabulary is dropped before the column order is frozen into gene_ids

    # Normally, this is not necessary, cause the h5ad is already filtered
    # to scGPT genes, is just for safety, this would print every gene of the h5ad
    in_vocab = np.array([g in vocab for g in adata.var.index])
    adata = adata[:, in_vocab].copy()
    genes = adata.var.index.tolist()
    print(f"Genes in vocab: {len(genes)}")

    # gene_ids is positional: gene_ids[j] is the token of column j of X, which
    # is the alignment tokenize_and_pad_batch relies on.
    gene_ids = np.array(vocab(genes), dtype=int)

    # The h5ad is already normalised and log1p'd upstream, so the only
    # transform left is the per-cell binning scGPT was pretrained on.
    pre = Preprocessor(
        use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
        normalize_total=False, log1p=False, subset_hvg=False,
        binning=C.N_BINS, result_binned_key="X_binned",
    )
    # scGPT's _digitize breaks bin ties with np.random.rand, and pad_batch keeps
    # a random MAX_SEQ_LEN - 1 genes of a cell with more expressed genes, so
    # without a fixed seed the tokens (and every downstream score) depend on the
    # global RNG state, i.e. on which cell types were processed before. The
    # global state is restored afterwards so other random steps are unaffected.
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        # The bins (0 to 50) fit in int8. Cells are binned one after another, so
        # the random tie-breaking draws are the same as binning all of them at once
        counts = np.zeros(adata.shape, dtype=np.int8)
        for start in range(0, adata.n_obs, BIN_CHUNK):
            chunk = adata[start:start + BIN_CHUNK].copy()
            pre(chunk, batch_key=None)
            binned = chunk.layers["X_binned"]
            counts[start:start + BIN_CHUNK] = binned.toarray() if issparse(binned) else binned

        # Tokenize and pad the batch, appending a CLS token and ignoring the zero-gene
        tok = tokenize_and_pad_batch(
            counts, gene_ids, max_len=C.MAX_SEQ_LEN, vocab=vocab,
            pad_token=C.PAD_TOKEN, pad_value=C.PAD_VALUE,
            append_cls=True, include_zero_gene=False,
        )
    finally:
        np.random.set_state(rng_state)

    # Convert the diagnosis labels to 0/1
    labels = labels_to_int(adata.obs[C.LABEL_COL].values)
    return {
        "gene_ids":         tok["genes"],
        "values":           tok["values"].float(),
        "condition_labels": torch.from_numpy(labels).long(),
    }


def _slug_cell_type(cell_type: str) -> str:
    return "_".join(str(cell_type).strip().lower().split())


# scGPT zero-shot model load. The Step 2.5 model is the same backbone with
# other weights, so it is loaded the same way with its own weights_path
def _load_zero_shot_model(vocab, model_configs: dict, device: torch.device,
                          weights_path=C.SCGPT_WEIGHTS) -> nn.Module:
    model = build_base_model(vocab, model_configs, n_cls=2)
    load_pretrained_weights(model, device, weights_path)
    model.to(device); model.eval()
    return model


# Zero-shot model + Step 4 MLP head: the pretrained backbone, unchanged, with the
# head pipeline_unified Step 4 trained on its CLS embedding as cls_decoder
# (the head the LoRA was warm-started from, before any fine-tuning)
def _load_mlp_model(head_path, vocab, model_configs: dict,
                    device: torch.device) -> nn.Module:
    model = build_base_model(vocab, model_configs, n_cls=1)
    load_pretrained_weights(model, device)
    model.cls_decoder = FlexMLP(**C.SCGPT_LORA_HEAD_CONFIG)
    model.cls_decoder.load_state_dict(torch.load(head_path, map_location=device)["state_dict"])
    model.to(device); model.eval()
    return model


# LoRA model load: pipeline_unified Step 5.1 saves the adapters merged into
# the backbone and the head as cls_decoder, all in one plain state_dict
def _load_lora_model(full_model_path, vocab, model_configs: dict,
                     device: torch.device) -> nn.Module:
    model = build_base_model(vocab, model_configs, n_cls=1)
    model.cls_decoder = FlexMLP(**C.SCGPT_LORA_HEAD_CONFIG)
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    model.to(device); model.eval()
    return model

# Attention layer extraction
# This is necessary because the scGPT model expose the attention trough 
# FlashMHA which it not giving us the attention
# So be need the input embeddings of the last later (outputs of the second-to-last layer)
class _LayerInputCapture:
    """Forward pre-hook that captures the input tensor passed to a module."""

    def __init__(self, module: nn.Module):
        self.captured: Optional[torch.Tensor] = None
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if isinstance(args, tuple) and len(args):
            self.captured = args[0].detach()

    def remove(self):
        self._handle.remove()

# Average over head of the last layer
# cls_scores : (B, H, M) raw attention logits for the CLS row.
# pad_mask   : (B, M) True at padded positions.
# returns    : (B, M) head-averaged rank-normalised attention; pads -> 0.  
def _cls_attn_rank_norm_head_avg(
    cls_scores: torch.Tensor, pad_mask: torch.Tensor,
) -> torch.Tensor:
    # Attention dimensions
    # In theory, H should be 8, because there are 8 heads per layer
    B, H, M = cls_scores.shape

    # The unsqueeze is for expand all 8 heads into one vector
    mask_bhM = pad_mask.unsqueeze(1).expand_as(cls_scores)

    # Replace the padded positions with negative infinity
    safe = cls_scores.masked_fill(mask_bhM, float("-inf"))

    # Order and rank values of attention
    order = torch.argsort(safe, dim=-1)
    ranks = torch.argsort(order, dim=-1).float()

    # Number of pad and valid position per cell
    n_pad = pad_mask.sum(dim=-1, keepdim=True).float()
    n_valid = (~pad_mask).sum(dim=-1, keepdim=True).float()

    # This is for ignore padding positions in the average
    shifted = (ranks - n_pad.unsqueeze(1)) / n_valid.clamp(min=1.0).unsqueeze(1)
    shifted = shifted.masked_fill(mask_bhM, 0.0)

    # Return the average value of all heads ignoring padding
    return shifted.mean(dim=1)

# Average of attention per gene per condition
def _gene_scores(
    means: Dict[int, Dict[str, float]],
    score_fn: Callable[[float, float], float],
) -> Dict[str, float]:
    all_genes = set(means[0].keys()) | set(means[1].keys())
    scores: Dict[str, float] = {}
    for g in all_genes:
        scores[g] = float(score_fn(means[1].get(g, 0.0), means[0].get(g, 0.0)))
    return scores
