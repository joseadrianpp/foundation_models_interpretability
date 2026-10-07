"""
Shared logic for the attention-based and the IG interpretability task.

Pipeline:
  1. Tokenise input cells with `include_zero_gene=False`
  2. Forward-pass through the model, capturing the input to encoder layer 11
     via a forward pre-hook.
  3. Compute CLS-row attention scores from Q.Kt (per head), rank-normalise
     across the gene axis (ignoring padding), average across heads.
  4. Aggregate per-gene CLS attention per condition. For the LoRA model we
     restrict to correctly classified cells.
  5. Score each gene with a user-supplied `score_fn(mean_pd, mean_ctrl)`.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from einops import rearrange
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
def _tokenise(adata: sc.AnnData, vocab) -> Dict[str, torch.Tensor]:
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
    # The bins (0 to 50) fit in int8. Cells are still binned one after another,
    # so the random tie-breaking draws are the same as binning all of them at once
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

    # Convert the diagnosis labels to 0/1
    labels = labels_to_int(adata.obs[C.LABEL_COL].values)
    return {
        "gene_ids":         tok["genes"],
        "values":           tok["values"].float(),
        "condition_labels": torch.from_numpy(labels).long(),
    }


def _slug_cell_type(cell_type: str) -> str:
    return "_".join(str(cell_type).strip().lower().split())


# scGPT zero-shot model load
def _load_zero_shot_model(vocab, model_configs: dict, device: torch.device) -> nn.Module:
    model = build_base_model(vocab, model_configs, n_cls=2)
    load_pretrained_weights(model, device)
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

# Forward pass to compute attention for the CLS token
def _forward_cls_attention(
    model: nn.Module, batch_pt: Dict[str, torch.Tensor],
    vocab, n_head: int, device: torch.device, n_cls: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # The token and gene ids and values
    pad_id = vocab[C.PAD_TOKEN]
    gids = batch_pt["gene_ids"].to(device)
    vals = batch_pt["values"].to(device)

    # True to padded position (not all cells are equal length)
    mask = gids.eq(pad_id)

    capture = _LayerInputCapture(model.transformer_encoder.layers[NUM_ATTN_LAYERS])
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(gids, vals, src_key_padding_mask=mask, batch_labels=None,
                           CLS=True, CCE=False, MVC=False, ECS=False)
    finally:
        capture.remove()
    embs = capture.captured.float()

    # FlashMHA of the last encoder layer
    self_attn = model.transformer_encoder.layers[NUM_ATTN_LAYERS].self_attn

    # Compute Q, K and V
    qkv = self_attn.Wqkv(embs)
    # Separate the 1536 dimension of QKV to individual variables
    # Index 0 if Q
    # Index 1 is K
    # Index 2 is V
    qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=n_head)
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]

    # The first token of Q is the CLS (0:1)
    q_cls = q[:, 0:1, :, :].permute(0, 2, 1, 3)

    # Transpose of K
    k_t = k.permute(0, 2, 3, 1)

    # Attention = Q*K^T / sqrt(d_k)
    cls_scores = (q_cls @ k_t).squeeze(2)

    # AVERAGE the attention scores
    cls_attn = _cls_attn_rank_norm_head_avg(cls_scores, mask)

    # We need the final CLS output because we need the CORRECT CLASSIFIED cells 
    cls_output = output["cls_output"].float()
    if n_cls == 1:
        logits = cls_output.squeeze(1)
        preds = (logits > 0).long()
    else:
        preds = cls_output.argmax(dim=1)
    return cls_attn, preds

# Caculate the attention between the CLS token and the gene tokens
def _compute_attn_per_gene_per_condition(
    model: nn.Module, pt: Dict[str, torch.Tensor], vocab, gene_names: dict, n_head: int,
    device: torch.device, n_cls: int, use_predictions: bool, batch_size: int = 16,
) -> Tuple[Dict[int, Dict[str, float]], dict]:

    # number of cells and special tokens
    n_cells = pt["gene_ids"].shape[0]
    pad_id = vocab[C.PAD_TOKEN]
    special_ids = {vocab[t] for t in C.SPECIAL_TOKENS if t in vocab}

    # Initialize variables to save attention values
    sum_by_gid: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
    n_per_cond = {0: 0, 1: 0}
    n_correct = {0: 0, 1: 0}

    # Iterate over batches of cells
    for start in range(0, n_cells, batch_size):

        # Check special case for last batch
        end = min(start + batch_size, n_cells)
        batch = {k: v[start:end] for k, v in pt.items()}

        # Calculate the attention AVERAGE over the 8 heads
        cls_attn, preds = _forward_cls_attention(
            model, batch, vocab, n_head, device, n_cls,
        )

        # Genes to tensors for gene ids and labels (which we don't currently have)
        gids = batch["gene_ids"].numpy()
        labels = batch["condition_labels"].numpy()
        # This is the prediction, which is in GPU because of the heads attention
        preds_np = preds.cpu().numpy()
        cls_attn_np = cls_attn.cpu().numpy()

        # This bucle is cell by cell
        for i in range(end - start):

            # True and pred labels
            true = int(labels[i])
            pred = int(preds_np[i])
            n_per_cond[true] += 1
            # If the cell is correctly classified, we count it for the attention
            if pred == true:
                n_correct[true] += 1
            if use_predictions and pred != true:
                continue
            # This is the attention for the i cell
            row = cls_attn_np[i]
            cell_gids = gids[i]

            # This bucle is gene by gene, because the attention we want is 
            # the attention of genes, not cells
            for pos in range(1, len(cell_gids)):
                gid = int(cell_gids[pos])
                if gid == pad_id or gid in special_ids:
                    continue
                sum_by_gid[true][gid] = sum_by_gid[true].get(gid, 0.0) + float(row[pos])

    # Cells that entered each condition: all of them, or only the correctly classified ones
    n_used = n_correct if use_predictions else n_per_cond

    # Compute the mean attention per gene per condition (positive; PD and negative; Control).
    # The mean is over every cell of the condition and a cell without the gene adds 0,
    # so a gene seen in a handful of cells cannot outrank one seen in all of them
    means: Dict[int, Dict[str, float]] = {0: {}, 1: {}}
    for cond in (0, 1):
        for gid, s in sum_by_gid[cond].items():
            # The current gene symbol, not the vocabulary's (GBA1, not GBA)
            name = gene_names[gid]
            if name in C.SPECIAL_TOKENS:
                continue
            means[cond][name] = s / n_used[cond] if n_used[cond] > 0 else 0.0

    # Safe case for 100% or 0% accuracy
    if use_predictions:
        ok = n_correct[0] > 0 and n_correct[1] > 0
        reason = "" if ok else (
            f"correctly-classified counts: control={n_correct[0]}, pd={n_correct[1]} - "
            f"at least one condition has 0 correct cells"
        )
    else:
        ok = n_per_cond[0] > 0 and n_per_cond[1] > 0
        reason = "" if ok else (
            f"ground-truth counts: control={n_per_cond[0]}, pd={n_per_cond[1]}"
        )
    gate = {"ok": bool(ok), "reason": reason, "mode": "predictions" if use_predictions else "labels"}
    return means, gate


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
