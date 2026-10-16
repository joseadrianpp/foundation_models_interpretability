"""
The LoRA scGPT classifier and its per-cell explanations, in the form Quantus
works with (the counterpart of rank_model.py).

scGPT reads a cell as (gene token, binned expression value) pairs, so:
  - x is the cell's expression values (shape (B, 1, L), one cell per
    evaluation, without padding); the cell's gene tokens are fixed by
    `set_cell`. Removing a gene is setting its value to 0, Quantus'
    `perturb_baseline`, the same baseline as the pipeline's IG
    (step7_explainability_ig_scgpt.py). Position 0 is the <cls> token, whose
    value the wrapper keeps at 0.
  - The two-class logits are (-z/2, z/2) of the PD logit z, as for the rank
    models.

Explanations, one value per position, 0 on <cls>:
  - ig / ig_abs: Integrated Gradients of the target logit w.r.t. the values
    (the pipeline's IG of z, times +1/2 for PD and -1/2 for control), and |ig|.
  - attention: the pipeline's CLS attention of layer 11 (rank-normalised,
    averaged over heads), the same for both targets.
  - random: uniform noise.
"""

import numpy as np
import torch
from torch import nn

from .. import config as C
from ..steps import attention_cssi_leiden as cssi
from ..steps import step7_explainability_ig_scgpt as ig
from .rank_model import explain_random


class ScGPTValueClassifier(nn.Module):
    """The LoRA scGPT classifier taking one cell's expression values as input."""

    remove_value = 0.0

    def __init__(self, net: nn.Module, vocab, n_head: int):
        super().__init__()
        self.net = net
        self.vocab = vocab
        self.n_head = n_head
        self.pad_id = vocab[C.PAD_TOKEN]
        self.special_pos = np.array([0])

    def set_cell(self, gene_ids: np.ndarray) -> None:
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


def _zero_cls(a: np.ndarray) -> np.ndarray:
    a[:, 0] = 0.0
    return a


def explain_ig(model: ScGPTValueClassifier, inputs, targets, n_steps: int = ig.IG_STEPS, **kwargs) -> np.ndarray:
    """Signed IG of the target logit for every gene value of every input (B, 1, L)."""
    vals = model.values(inputs)
    gids = model.gids(len(vals))
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(False)
        ig_z = ig._ig_batch(model.net, gids, vals, gids.eq(model.pad_id), model.device, n_steps, [len(vals)])
    sign = np.where(np.asarray(targets).reshape(-1) == 1, 0.5, -0.5)[:, None]
    return _zero_cls(ig_z * sign).reshape(np.shape(inputs)).astype(np.float32)


def explain_ig_abs(model: ScGPTValueClassifier, inputs, targets, n_steps: int = ig.IG_STEPS, **kwargs) -> np.ndarray:
    return np.abs(explain_ig(model, inputs, targets, n_steps))


def explain_attention(model: ScGPTValueClassifier, inputs, targets, **kwargs) -> np.ndarray:
    """CLS attention of every position of every input (B, 1, L), as the pipeline computes it."""
    vals = model.values(inputs).cpu()
    batch = {"gene_ids": model.gene_ids.expand(len(vals), -1), "values": vals}
    cls_attn, _, _ = cssi._forward_cls_attention_with_embedding(model.net, batch, model.vocab, model.n_head,
                                                                model.device, n_cls=1)
    return _zero_cls(cls_attn.float().cpu().numpy()).reshape(np.shape(inputs)).astype(np.float32)


EXPLAIN_FUNCS = {"ig": explain_ig, "ig_abs": explain_ig_abs, "attention": explain_attention,
                 "random": explain_random}
