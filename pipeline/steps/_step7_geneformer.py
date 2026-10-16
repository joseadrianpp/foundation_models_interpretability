"""
Attention-based interpretability for Geneformer: the same CLS attention per
condition as the scGPT attention task (steps/_step7_common.py). scCello runs
through the same code: it is also a BERT over rank value encoded cells, only
smaller, with its own vocabulary and its cell embedding in the last block.

Pipeline:
  1. The cells come already tokenized (rank value encoding, <cls> first and,
     in Geneformer, <eos> last), so each batch only has to be padded to its
     longest cell.
  2. Forward-pass through the model, capturing the input to the encoder block
     that outputs the cell embedding (Geneformer: 10 of 12, scCello: 5 of 6)
     via a forward pre-hook.
  3. Compute CLS-row attention scores from Q.Kt (per head), rank-normalise
     across the gene axis (ignoring padding), average across heads.
  4. Aggregate per-gene CLS attention per condition (attention_cssi_leiden.py).
     For the LoRA model we restrict to correctly classified cells.

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


# Geneformer or scCello zero-shot model load, only the encoder since attention needs no head.
# The Step 2.5 model is the same encoder with other weights, loaded from its own folder
def _load_zero_shot_model(model_dir, device: torch.device) -> nn.Module:
    model = BertModel.from_pretrained(str(model_dir), add_pooling_layer=False)
    model.to(device); model.eval()
    return model


# LoRA model load: the adapter was trained on a BertForSequenceClassification
# (scCello: on its own subclass of it, with the same module names), so peft needs
# that same model underneath (its own pooler and classifier are never used); the
# head is the Step 4 FlexMLP
def _load_lora_model(lora_dir, model_dir, head_config: dict, emb_layer: int,
                     device: torch.device) -> nn.Module:
    backbone = BertForSequenceClassification.from_pretrained(str(model_dir), num_labels=2)
    backbone = PeftModel.from_pretrained(backbone, str(lora_dir / "best_lora"))
    head = FlexMLP(**head_config)
    head.load_state_dict(torch.load(lora_dir / "best_head.pt", map_location=device))
    model = GeneformerClassifier(backbone, head, emb_layer)
    model.to(device); model.eval()
    return model


# Zero-shot model + Step 4 MLP head: the same classifier as the LoRA one without
# the adapters, i.e. the pretrained encoder with the head pipeline_unified Step 4
# trained on its zero-shot cell embedding (the LoRA's starting point)
def _load_mlp_model(head_path, model_dir, head_config: dict, emb_layer: int,
                    device: torch.device) -> nn.Module:
    backbone = BertForSequenceClassification.from_pretrained(str(model_dir), num_labels=2)
    head = FlexMLP(**head_config)
    head.load_state_dict(torch.load(head_path, map_location=device)["state_dict"])
    model = GeneformerClassifier(backbone, head, emb_layer)
    model.to(device); model.eval()
    return model


# The BERT of a zero-shot model (the model itself) or of a LoRA classifier
def _bert(model: nn.Module) -> nn.Module:
    return model.backbone.bert if isinstance(model, GeneformerClassifier) else model


# 0-indexed encoder block whose output is the cell embedding (hidden_states[emb_layer]
# is the output of encoder.layer[emb_layer]): Geneformer 10 of 12, scCello 5 of 6.
# Its attention is the last one that shapes what the head classifies
def attn_layer_index(model: nn.Module, emb_layer: int) -> int:
    return emb_layer % _bert(model).config.num_hidden_layers


# CLS attention of the attention block `layer_idx`, the predictions of a LoRA
# classifier (use_predictions) and the residual-stream CLS embedding entering
# that block (the cell embedding the CSSI strata are built from, see
# attention_cssi_leiden.collect_geneformer)
def _forward_cls_attention_with_embedding(
    model: nn.Module, ids: torch.Tensor, attn: torch.Tensor, layer_idx: int, use_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # The zero-shot model is the BERT itself, the LoRA one has it inside the classifier
    bert = _bert(model)
    layer = bert.encoder.layer[layer_idx]
    n_head = bert.config.num_attention_heads
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
        # Rank-normalised and AVERAGED over the heads (12 in Geneformer, 4 in scCello)
        cls_attn = _cls_attn_rank_norm_head_avg(cls_scores, mask)

    preds = (logits > 0).long() if use_predictions else None
    return cls_attn, preds, embs[:, 0, :]
