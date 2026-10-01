"""
Attention-based interpretability for Geneformer: the same CLS attention per
condition as the scGPT attention task (steps/_step7_common.py).

Pipeline:
  1. The cells come already tokenized (rank value encoding, <cls> first and
     <eos> last), so each batch only has to be padded to its longest cell.
  2. Forward-pass through the model, capturing the input to encoder layer 10
     via a forward pre-hook.
  3. Compute CLS-row attention scores from Q.Kt (per head), rank-normalise
     across the gene axis (ignoring padding), average across heads.
  4. Aggregate per-gene CLS attention per condition. For the LoRA model we
     restrict to correctly classified cells.

The per-gene score (PD - Control) and the top-N file are the same as for
scGPT (run_tasks.py).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from peft import PeftModel
from torch import nn
from transformers import BertForSequenceClassification, BertModel

from .. import config as C
from ..utils.geneformer_model import GeneformerClassifier
from ..utils.scgpt_model import FlexMLP
from ..utils.splits import labels_to_int
from ._step7_common import _LayerInputCapture, _cls_attn_rank_norm_head_avg

# 0-indexed transformer encoder layer for attention. Geneformer's cell embedding
# is the output of the second-to-last layer (EMB_HIDDEN_LAYER = -2), so layer
# 10 of 12 is the last one that shapes what the head classifies
NUM_ATTN_LAYER = 10

# Geneformer cells have up to 4096 tokens, so the batch is smaller than scGPT's
ATTN_BATCH_SIZE = 8


# From the dataset to what the model needs: token ids of every cell and 0/1 labels
def _to_inputs(ds) -> Dict:
    ds = ds.with_format("numpy")
    labels = labels_to_int(ds[C.LABEL_COL])
    return {
        "input_ids": [np.asarray(ids, dtype=np.int64) for ids in ds["input_ids"]],
        "condition_labels": torch.from_numpy(labels).long(),
    }


# Cells have different lengths, so each batch is padded to its longest cell.
# The attention mask is 1 on real tokens and 0 on padding
def _pad_batch(seqs: List[np.ndarray], pad_id: int,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.from_numpy(s)
        attn[i, :len(s)] = 1
    return ids.to(device), attn.to(device)


# Geneformer zero-shot model load, only the encoder since attention needs no head
def _load_zero_shot_model(device: torch.device) -> nn.Module:
    model = BertModel.from_pretrained(str(C.GENEFORMER_MODEL_DIR), add_pooling_layer=False)
    model.to(device); model.eval()
    return model


# LoRA model load: the adapter was trained on a BertForSequenceClassification,
# so peft needs that same model underneath (its own pooler and classifier are
# never used); the head is the Step 4 FlexMLP
def _load_lora_model(lora_dir, device: torch.device) -> nn.Module:
    backbone = BertForSequenceClassification.from_pretrained(
        str(C.GENEFORMER_MODEL_DIR), num_labels=2)
    backbone = PeftModel.from_pretrained(backbone, str(lora_dir / "best_lora"))
    head = FlexMLP(**C.GENEFORMER_LORA_HEAD_CONFIG)
    head.load_state_dict(torch.load(lora_dir / "best_head.pt", map_location=device))
    model = GeneformerClassifier(backbone, head)
    model.to(device); model.eval()
    return model


# Forward pass to compute attention for the CLS token
def _forward_cls_attention(
    model: nn.Module, ids: torch.Tensor, attn: torch.Tensor, use_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # The zero-shot model is the BERT itself, the LoRA one has it inside the classifier
    bert = model.backbone.bert if isinstance(model, GeneformerClassifier) else model
    layer = bert.encoder.layer[NUM_ATTN_LAYER]
    n_head = bert.config.num_attention_heads

    # True to padded positions (not all cells are equal length)
    mask = attn.eq(0)

    capture = _LayerInputCapture(layer)
    try:
        with torch.no_grad():
            # The LoRA forward also returns the PD logit, needed for the correctly classified cells
            if use_predictions:
                logits = model(input_ids=ids, attention_mask=attn)
            else:
                bert(input_ids=ids, attention_mask=attn)
    finally:
        capture.remove()

    with torch.no_grad():
        embs = capture.captured.float()

        # Unlike scGPT's FlashMHA, the BERT layer has separate Q and K projections
        self_attn = layer.attention.self
        q = rearrange(self_attn.query(embs), "b s (h d) -> b h s d", h=n_head)
        k = rearrange(self_attn.key(embs), "b s (h d) -> b h s d", h=n_head)

        # Attention of the CLS (first token of Q) to every token: Q_cls*K^T.
        # Dividing by sqrt(d_k) is left out because it does not change the ranks
        cls_scores = (q[:, :, 0:1, :] @ k.transpose(-1, -2)).squeeze(2)

        # AVERAGE the attention scores
        cls_attn = _cls_attn_rank_norm_head_avg(cls_scores, mask)

    preds = (logits > 0).long() if use_predictions else None
    return cls_attn, preds


# Add the attention of one cell to the per-gene sums, gene by gene, because the
# attention we want is the attention of genes, not cells
def _add_cell(sums: Dict[int, float], cell_gids: np.ndarray, row: np.ndarray,
              special_ids: set) -> None:
    for pos in range(1, len(cell_gids)):
        gid = int(cell_gids[pos])
        if gid in special_ids:
            continue
        sums[gid] = sums.get(gid, 0.0) + float(row[pos])


# Calculate the attention between the CLS token and the gene tokens
def _compute_attn_per_gene_per_condition(
    model: nn.Module, pt: Dict, token_dict: dict, gene_names: dict,
    device: torch.device, use_predictions: bool, batch_size: int = ATTN_BATCH_SIZE,
) -> Tuple[Dict[int, Dict[str, float]], dict]:

    n_cells = len(pt["input_ids"])
    pad_id = token_dict["<pad>"]
    # <pad>, <mask>, <cls> and <eos> are not genes
    special_ids = {v for k, v in token_dict.items() if k.startswith("<")}

    # Initialize variables to save attention values
    sum_by_gid: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
    n_used = {0: 0, 1: 0}

    # Iterate over batches of cells
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        ids, attn = _pad_batch(pt["input_ids"][start:end], pad_id, device)

        # Calculate the attention AVERAGE over the 12 heads
        cls_attn, preds = _forward_cls_attention(model, ids, attn, use_predictions)

        gids = ids.cpu().numpy()
        labels = pt["condition_labels"][start:end].numpy()
        cls_attn_np = cls_attn.cpu().numpy()

        # This bucle is cell by cell
        for i in range(end - start):
            true = int(labels[i])
            # With the LoRA model only the correctly classified cells count
            if use_predictions and int(preds[i]) != true:
                continue
            n_used[true] += 1
            _add_cell(sum_by_gid[true], gids[i], cls_attn_np[i], special_ids)

    # Compute the mean attention per gene per condition (positive; PD and negative; Control).
    # The mean is over every cell of the condition and a cell without the gene adds 0,
    # so a gene seen in a handful of cells cannot outrank one seen in all of them
    means: Dict[int, Dict[str, float]] = {0: {}, 1: {}}
    for cond in (0, 1):
        for gid, s in sum_by_gid[cond].items():
            means[cond][gene_names[gid]] = s / n_used[cond]

    # Safe case for a condition without cells (e.g. 0% accuracy on it)
    ok = n_used[0] > 0 and n_used[1] > 0
    reason = "" if ok else f"cells used: control={n_used[0]}, pd={n_used[1]}"
    gate = {"ok": bool(ok), "reason": reason, "mode": "predictions" if use_predictions else "labels"}
    return means, gate
