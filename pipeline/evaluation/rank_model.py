"""
The LoRA classifiers of the rank-value encoded models (Geneformer, scCello)
and their per-cell explanations, in the form Quantus works with.

Quantus perturbs a numeric input array x and reads the model's class
outputs, so:
  - x is the cell itself, the token id of every position (shape (B, 1, L),
    one cell per evaluation: cells have different lengths, and padding would
    be perturbed as if it were genes). Removing a gene is replacing its id
    by <mask>, Quantus' `perturb_baseline`, the same baseline as our
    Integrated Gradients (step7_explainability_ig_geneformer.py). The
    special tokens (<cls> first, Geneformer's <eos> last) are put back by
    the wrapper, so perturbing them changes nothing.
  - The classifier has one PD logit z; the wrapper returns the two-class
    logits (-z/2, z/2), whose softmax is (1 - sigmoid(z), sigmoid(z)), so
    the target is the cell's label as in any classifier.

Explanations (Quantus `explain_func`), one value per position, 0 on the
special tokens:
  - ig: Integrated Gradients of the target logit w.r.t. the token embeddings
    (the pipeline's IG of z, times +1/2 for PD and -1/2 for control), signed.
  - ig_abs: |ig|, the per-cell quantity the pipeline's IG ranking averages.
  - attention: the CLS attention of the block that outputs the cell
    embedding, rank-normalised and averaged over heads (the pipeline's
    attention), the same for both targets.
  - random: uniform noise, the reference every method should beat.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

from ..steps import _step7_geneformer as gf
from ..steps import step7_explainability_ig_geneformer as ig_gf
from ..steps.step7_explainability_ig_scgpt import IG_STEPS


class RankTokenClassifier(nn.Module):
    """A LoRA GeneformerClassifier (Geneformer or scCello) taking a cell's token ids as floats."""

    def __init__(self, classifier: nn.Module, token_dict: dict, emb_layer: int):
        super().__init__()
        self.classifier = classifier
        self.mask_id = int(token_dict["<mask>"])
        self.special_ids = np.array(sorted(v for k, v in token_dict.items() if k.startswith("<")))
        self.layer_idx = gf.attn_layer_index(classifier, emb_layer)
        self.cell: Optional[np.ndarray] = None

    @property
    def remove_value(self) -> int:
        """What a removed gene becomes (Quantus' perturb_baseline): the <mask> token."""
        return self.mask_id

    def set_cell(self, ids: np.ndarray) -> None:
        """The cell being explained: its special tokens are restored in every perturbed copy."""
        self.cell = np.asarray(ids, dtype=np.int64)
        self.special_pos = np.where(np.isin(self.cell, self.special_ids))[0]

    def token_ids(self, x) -> torch.Tensor:
        ids = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(len(x), -1).round().long()
        ids[:, self.special_pos] = torch.from_numpy(self.cell[self.special_pos])
        return ids.to(self.device)

    @property
    def device(self) -> torch.device:
        return next(self.classifier.parameters()).device

    def pd_logit(self, ids: torch.Tensor) -> torch.Tensor:
        return self.classifier(input_ids=ids, attention_mask=torch.ones_like(ids))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pd_logit(self.token_ids(x.detach().cpu().numpy()))
        return torch.stack([-z / 2, z / 2], dim=1)


def _zero_specials(model: RankTokenClassifier, a: np.ndarray) -> np.ndarray:
    a[:, model.special_pos] = 0.0
    return a


def explain_ig(model: RankTokenClassifier, inputs, targets, n_steps: int = IG_STEPS, **kwargs) -> np.ndarray:
    """Signed IG of the target logit for every position of every input (B, 1, L)."""
    ids = model.token_ids(inputs)
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(False)
        ig_z = ig_gf._ig_batch(model.classifier, ids, torch.ones_like(ids), model.mask_id, model.device,
                               n_steps, [len(ids)])
    sign = np.where(np.asarray(targets).reshape(-1) == 1, 0.5, -0.5)[:, None]
    return _zero_specials(model, ig_z * sign).reshape(np.shape(inputs)).astype(np.float32)


def explain_ig_abs(model: RankTokenClassifier, inputs, targets, n_steps: int = IG_STEPS, **kwargs) -> np.ndarray:
    """|IG| of every position: how much each gene moves the logit, whatever the direction."""
    return np.abs(explain_ig(model, inputs, targets, n_steps))


def explain_attention(model: RankTokenClassifier, inputs, targets, **kwargs) -> np.ndarray:
    """CLS attention of every position of every input (B, 1, L), as the pipeline computes it."""
    ids = model.token_ids(inputs)
    cls_attn, _, _ = gf._forward_cls_attention_with_embedding(model.classifier, ids, torch.ones_like(ids),
                                                              model.layer_idx, use_predictions=False)
    return _zero_specials(model, cls_attn.float().cpu().numpy()).reshape(np.shape(inputs)).astype(np.float32)


def explain_random(model: RankTokenClassifier, inputs, targets, **kwargs) -> np.ndarray:
    """Uniform noise (the global numpy RNG, seeded by the caller): the reference explanation."""
    a = np.random.uniform(size=np.shape(inputs)).reshape(len(inputs), -1).astype(np.float32)
    return _zero_specials(model, a).reshape(np.shape(inputs))


EXPLAIN_FUNCS = {"ig": explain_ig, "ig_abs": explain_ig_abs, "attention": explain_attention,
                 "random": explain_random}


def mask_tokens(arr: np.ndarray, indices: np.ndarray, mask_id: int, fraction: float = 0.02,
                **kwargs) -> np.ndarray:
    """A small discrete perturbation of token inputs: a random `fraction` of the positions become <mask>.

    The robustness metrics' default perturbation adds continuous noise to x,
    which on token ids would swap genes for arbitrary ones. Same signature as
    Quantus' batch perturbation functions (arr and indices are (B, features)).
    """
    out = arr.copy()
    for b in range(arr.shape[0]):
        n = max(1, int(round(fraction * indices.shape[1])))
        out[b, np.random.choice(indices[b], n, replace=False)] = mask_id
    return out
