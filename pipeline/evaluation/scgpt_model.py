"""
The scGPT models and their per-cell explanations, in the form Quantus works
with (the counterpart of rank_model.py).

scGPT reads a cell as (gene token, binned expression value) pairs, so:
  - x is the cell's expression values (shape (B, 1, L), one cell per
    evaluation, without padding); the cell's gene tokens are fixed by
    `set_cell`. Removing a gene is setting its value to 0, Quantus'
    `perturb_baseline`, the same baseline as the pipeline's IG
    (step7_explainability_ig_scgpt.py). Position 0 is the <cls> token, whose
    value the wrappers keep at 0.
  - The classifiers (zero-shot + MLP head, LoRA; ScGPTValueClassifier) have
    one PD logit z: the two-class logits are (-z/2, z/2), whose softmax is
    (1 - sigmoid(z), sigmoid(z)), as for the rank models. Quantus reads the
    target probability (softmax).
  - The encoders without a head (zero-shot, Step 2.5; ScGPTEmbeddingScorer)
    have no prediction: their output is the one their IG explains,
    SIGnature's y(x') = sum_k |w_k * f_k(x')| with w = f(x) the embedding of
    the real cell, rescaled to (y(x') - y(0)) / (y(x) - y(0)), with 0 the IG
    baseline (every value removed): it goes from 1 (the real cell) to 0 (every
    gene removed), the share of the output the removed genes account for. y
    itself barely moves (y(x) - y(0) is ~1 % of y(x) in scGPT), so y / y(x)
    would keep every deletion curve near 1. Both "classes" are that same
    value, read without softmax.

Explanations, one value per position, 0 on <cls>:
  - ig: signed Integrated Gradients. Classifiers: of the target logit (the
    pipeline's IG of z, times +1/2 for PD and -1/2 for control). Encoders:
    of SIGnature's output (the pipeline's per-cell IG before its scaling to
    1000, which does not change the order of a cell's genes). The absolute
    value is not taken: |IG| scored far lower FG (evaluate_attributions.py).
  - attention: the pipeline's CLS attention of layer 11 (rank-normalised,
    averaged over heads), the same for both targets.
  - random: uniform noise.
  - ranking_explainer(): a population ranking of the pipeline (pooled, CSSI
    or OLS of score IG or IG±) as every cell's attribution (IG± flipped for
    control cells only for the encoders).
"""

import numpy as np
import torch
from torch import nn

from .. import config as C
from ..steps import attention_cssi_leiden as cssi
from ..steps import step7_explainability_ig_scgpt as ig
from .rank_model import explain_random


class ScGPTValueClassifier(nn.Module):
    """A scGPT PD classifier (zero-shot + MLP head or LoRA) taking one cell's expression values as input."""

    remove_value = 0.0
    softmax = True
    n_cls = 1
    output = "pd_logit"

    def __init__(self, net: nn.Module, vocab, n_head: int):
        super().__init__()
        self.net = net
        self.vocab = vocab
        self.n_head = n_head
        self.pad_id = vocab[C.PAD_TOKEN]
        self.special_pos = np.array([0])

    def set_cell(self, gene_ids: np.ndarray, x=None) -> None:
        """The gene tokens of the cell being explained (<cls> first, no padding)."""
        self.gene_ids = torch.as_tensor(np.asarray(gene_ids), dtype=torch.long)[None, :]

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    def values(self, x) -> torch.Tensor:
        vals = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(len(x), -1).clone()
        vals[:, 0] = 0.0
        return vals.to(self.device)

    def gids(self, n: int) -> torch.Tensor:
        return self.gene_ids.expand(n, -1).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = self.values(x.detach().cpu().numpy())
        gids = self.gids(len(vals))
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            z = ig._pd_logit(self.net, gids, vals, gids.eq(self.pad_id)).float()
        return torch.stack([-z / 2, z / 2], dim=1)


class ScGPTEmbeddingScorer(ScGPTValueClassifier):
    """A scGPT encoder without a head (zero-shot or Step 2.5), scored by SIGnature's output."""

    softmax = False
    n_cls = 2
    output = "signature"

    def set_cell(self, gene_ids: np.ndarray, x=None) -> None:
        """The cell being explained: its tokens, its embedding w, and y(x) and y(0), fixed."""
        super().set_cell(gene_ids)
        vals = self.values(x)
        self.weights = None
        with torch.no_grad():
            self.weights = self._embedding(vals)
            self.y_real = float(self._signature(vals))
            self.y_base = float(self._signature(torch.zeros_like(vals)))

    def _embedding(self, vals: torch.Tensor) -> torch.Tensor:
        gids = self.gids(len(vals))
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            return ig._cell_embedding(self.net, gids, vals, gids.eq(self.pad_id)).float()

    def _signature(self, vals: torch.Tensor) -> torch.Tensor:
        gids = self.gids(len(vals))
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            return ig._signature_output(self.net, gids, vals, gids.eq(self.pad_id),
                                        self.weights.expand(len(vals), -1)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = self.values(x.detach().cpu().numpy())
        with torch.no_grad():
            y = (self._signature(vals) - self.y_base) / (self.y_real - self.y_base)
        return torch.stack([y, y], dim=1)


def _zero_cls(a: np.ndarray) -> np.ndarray:
    a[:, 0] = 0.0
    return a


def explain_ig(model: ScGPTValueClassifier, inputs, targets, n_steps: int = ig.IG_STEPS, **kwargs) -> np.ndarray:
    """Signed IG for every gene value of every input (B, 1, L): of the target logit, or of SIGnature's output."""
    vals = model.values(inputs)
    gids = model.gids(len(vals))
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(False)
        a = ig._ig_batch(model.net, gids, vals, gids.eq(model.pad_id), model.device, n_steps, [len(vals)],
                         model.output)
    if model.output == "pd_logit":
        a = a * np.where(np.asarray(targets).reshape(-1) == 1, 0.5, -0.5)[:, None]
    return _zero_cls(a).reshape(np.shape(inputs)).astype(np.float32)


def explain_attention(model: ScGPTValueClassifier, inputs, targets, **kwargs) -> np.ndarray:
    """CLS attention of every position of every input (B, 1, L), as the pipeline computes it."""
    vals = model.values(inputs).cpu()
    batch = {"gene_ids": model.gene_ids.expand(len(vals), -1), "values": vals}
    cls_attn, _, _ = cssi._forward_cls_attention_with_embedding(model.net, batch, model.vocab, model.n_head,
                                                                model.device, n_cls=model.n_cls)
    return _zero_cls(cls_attn.float().cpu().numpy()).reshape(np.shape(inputs)).astype(np.float32)


def ranking_explainer(scores: dict, signed: bool):
    """A population ranking (token id -> score) as the attribution of every gene of a cell.

    Genes the ranking does not hold get 0. With `signed` (score IG± of an
    encoder, PD - control) the scores are flipped for control targets, so a
    cell's most relevant genes are those of its own condition. A classifier's
    IG± is already towards the true class of each condition: not signed.
    """
    def explain(model: ScGPTValueClassifier, inputs, targets, **kwargs) -> np.ndarray:
        gids = model.gene_ids[0].numpy()
        a = np.array([scores.get(int(g), 0.0) for g in gids], dtype=np.float32)[None, :]
        a = np.repeat(a, len(inputs), axis=0)
        if signed:
            a = a * np.where(np.asarray(targets).reshape(-1) == 1, 1.0, -1.0)[:, None]
        return _zero_cls(a).reshape(np.shape(inputs)).astype(np.float32)
    return explain


EXPLAIN_FUNCS = {"ig": explain_ig, "attention": explain_attention, "random": explain_random}
