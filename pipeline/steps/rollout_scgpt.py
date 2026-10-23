"""
Attention rollout (Abnar and Zuidema, ACL 2020) of scGPT's <cls> token.

In every encoder layer l, the attention matrix A_l (softmax(QK^T / sqrt(d)),
averaged over the heads, padded keys masked) is mixed with the residual
connection, 0.5 * A_l + 0.5 * I, and the layers are multiplied from the last
to the first: the <cls> row of

    R = (0.5 A_L + 0.5 I) ... (0.5 A_1 + 0.5 I)

is how much each input position reaches the <cls> output through every path.
Only that row is needed, so it is propagated as a vector (one vector-matrix
product per layer). scGPT's FlashMHA does not return its attention weights:
each layer's input is captured with a hook and its attention recomputed from
the layer's own Wqkv (with LoRA's adapters when the model has them), as the
attention task does for its last layer.

Rollout follows the encoder only, not the head: the zero-shot and zero-shot +
MLP forms (same encoder) have the same rollout. It is non-negative and has no
class: a cell's rollout over its gene tokens is scaled to sum 1.

The per-cell values go through the IG outputs (run_tasks._write_ig_outputs):
score `rollout` (mean over the cells, as IG) and `rollout±` (PD - control
mean, as IG±), with the same rankings and selections.

Gradient-weighted rollout (Chefer, Gur and Wolf, ICCV 2021, "Generic
attention-model explainability"): each layer's attention weighted by the
gradient of the model's output with respect to it, positive part, averaged
over the heads, Ā_l = E_h[(∇A_l ⊙ A_l)^+], accumulated from the first layer
to the last, R = I, R <- R + Ā_l R; the <cls> row of R over the genes. The
output is the one the form's IG explains: the PD logit towards the true
class (classifiers: +z for PD cells, -z for control cells) or SIGnature's
sum_k |w_k f_k(x)| (encoders). Unlike rollout it depends on the head and on
the class. The attention needs gradients, so FlashMHA is replaced, while it
runs, by the same attention computed explicitly (_explicit_attention). Scores
`gradrollout` and `gradrollout±`, written like the rollout ones.
"""

from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange

from .. import config as C
from . import _step7_common as common
from . import step7_explainability_ig_scgpt as ig
from .step7_explainability_ig_scgpt import IGResult, run_ig

ROLLOUT_BATCH_SIZE = 8
GRAD_ROLLOUT_BATCH_SIZE = 4


def rollout_batch(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor,
                  n_head: int) -> torch.Tensor:
    """<cls> rollout of a batch of cells: (B, L), each row summing to 1 (0 on padding)."""
    layers = model.transformer_encoder.layers
    captures = [common._LayerInputCapture(layer) for layer in layers]
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(gids.device.type == "cuda")):
            model(gids, vals, src_key_padding_mask=mask, batch_labels=None,
                  CLS=False, CCE=False, MVC=False, ECS=False)
    finally:
        for c in captures:
            c.remove()

    v = torch.zeros(gids.shape, dtype=torch.float32, device=gids.device)
    v[:, 0] = 1.0
    pad = mask[:, None, None, :]
    with torch.no_grad():
        for layer, cap in zip(reversed(layers), reversed(captures)):
            x = cap.captured
            if getattr(layer, "norm_scheme", "post") == "pre":
                x = layer.norm1(x)
            with torch.cuda.amp.autocast(enabled=(gids.device.type == "cuda")):
                qkv = layer.self_attn.Wqkv(x)
            qkv = rearrange(qkv.float(), "b s (three h d) -> b s three h d", three=3, h=n_head)
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 3, 1)
            scores = (q @ k) * q.shape[-1] ** -0.5
            attn = torch.softmax(scores.masked_fill(pad, float("-inf")), dim=-1).mean(dim=1)
            del scores
            # Row vector times (0.5 A + 0.5 I)
            v = 0.5 * torch.bmm(v[:, None, :], attn)[:, 0] + 0.5 * v
    return v


class _explicit_attention:
    """Within the block, every layer's FlashMHA computes its attention explicitly and keeps the maps.

    softmax(QK^T / sqrt(d)) with the padded keys masked, then the same
    out_proj, so the layer's output is unchanged (up to precision); the maps
    (B, H, L, L) are in `self.maps`, in layer order, with their gradients kept.
    """

    def __init__(self, model: torch.nn.Module, n_head: int):
        self.layers = model.transformer_encoder.layers
        self.n_head = n_head
        self.maps = []

    def _forward(self, attn_module):
        def forward(x, key_padding_mask=None, need_weights=False, **kwargs):
            qkv = rearrange(attn_module.Wqkv(x), "b s (three h d) -> b s three h d", three=3, h=self.n_head)
            q, k, v = (qkv[:, :, j].transpose(1, 2) for j in range(3))
            scores = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
            if key_padding_mask is not None:
                # FlashMHA's mask is True on the positions to keep
                scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            if attn.requires_grad:
                attn.retain_grad()
            self.maps.append(attn)
            context = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            return attn_module.out_proj(context), None
        return forward

    def __enter__(self):
        for layer in self.layers:
            layer.self_attn.forward = self._forward(layer.self_attn)
        return self

    def __exit__(self, *exc):
        for layer in self.layers:
            del layer.self_attn.forward


def grad_rollout_batch(model: torch.nn.Module, gids: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor,
                       n_head: int, output: str, signs: torch.Tensor) -> torch.Tensor:
    """<cls> gradient-weighted rollout of a batch of cells: (B, L), 0 on padding (not normalised).

    `output`: "pd_logit" (the target is signs * z, signs = +1 for PD and -1
    for control cells) or "signature" (SIGnature's output of the real cell).
    """
    amp = gids.device.type == "cuda"
    vals = vals.clone().float().requires_grad_(True)
    with _explicit_attention(model, n_head) as ctx, torch.enable_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            if output == "signature":
                with torch.no_grad():
                    w = ig._cell_embedding(model, gids, vals.detach(), mask).float()
                ctx.maps.clear()
                target = ig._signature_output(model, gids, vals, mask, w).float()
            else:
                target = ig._pd_logit(model, gids, vals, mask).float() * signs
        target.sum().backward()
        maps = ctx.maps
    n = gids.shape[1]
    r = torch.eye(n, device=gids.device).expand(gids.shape[0], n, n).clone()
    with torch.no_grad():
        for attn in maps:
            cam = (attn.grad * attn).clamp(min=0).mean(dim=1)
            r = r + torch.bmm(cam, r)
    return r[:, 0].masked_fill(mask, 0.0)


def compute_rollout_result(model: torch.nn.Module, pt: Dict[str, torch.Tensor], cache, vocab, gene_names: dict,
                           device: torch.device, n_head: int, scored: np.ndarray,
                           batch_size: int = ROLLOUT_BATCH_SIZE, checkpoint=None,
                           desc: str = "rollout", output: Optional[str] = None) -> IGResult:
    """Rollout of every gene token of the `scored` cells, as an IGResult (token_ig = rollout).

    `cache` is the model's AttentionCache of the same cells (collect_scgpt),
    for the labels and the embeddings (Leiden strata of CSSI). With `output`
    ("pd_logit" or "signature") the gradient-weighted rollout of that output.
    """
    pad_id = vocab[C.PAD_TOKEN]
    special_arr = np.array(sorted({vocab[t] for t in C.SPECIAL_TOKENS if t in vocab} | {pad_id}))

    def fn(idx: np.ndarray):
        gids = pt["gene_ids"][idx].to(device)
        vals = pt["values"][idx].to(device).float()
        if output is None:
            r = rollout_batch(model, gids, vals, gids.eq(pad_id), n_head)
        else:
            signs = torch.as_tensor(np.where(cache.labels[idx] == 1, 1.0, -1.0), dtype=torch.float32, device=device)
            r = grad_rollout_batch(model, gids, vals, gids.eq(pad_id), n_head, output, signs)
        r = r.detach().cpu().numpy()
        gids_np = gids.cpu().numpy()
        keep = ~np.isin(gids_np, special_arr)
        keep[:, 0] = False
        # Each cell's rollout over its gene tokens (the <cls> keeps a share of its own)
        r = np.where(keep, r, 0.0)
        r = r / np.maximum(r.sum(axis=1, keepdims=True), 1e-12)
        return gids_np, r.astype(np.float32), keep

    make_batches = lambda order: [order[i:i + batch_size] for i in range(0, order.size, batch_size)]
    tokens = run_ig(np.where(scored)[0], make_batches, fn, checkpoint, desc)
    return IGResult.from_tokens(*tokens, embeddings=cache.embeddings, labels=cache.labels,
                                correct=np.asarray(scored, dtype=bool), names=gene_names)


def subset_result(result: IGResult, scored: np.ndarray, names: dict) -> IGResult:
    """The same rollout on fewer cells (zero-shot + MLP: the zero-shot rollout of its correct cells)."""
    scored = np.asarray(scored, dtype=bool) & result.correct
    keep = scored[result.token_cells]
    return IGResult.from_tokens(result.token_cells[keep], result.token_gids[keep], result.token_ig[keep],
                                embeddings=result.embeddings, labels=result.labels, correct=scored, names=names)
