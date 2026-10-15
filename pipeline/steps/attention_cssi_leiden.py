"""Paper-faithful CSSI for the gene-level zero-shot attention pipeline.

The paper's CSSI stratification is:
  1. model cell embeddings from the residual stream
  2. cosine k-NN graph (k=15 default)
  3. Leiden community detection, targeting ~5-7 strata
  4. score attention within each stratum
  5. CSSI-max aggregation across strata

Your existing output is gene-level (PD-control attention), not TF-target edges,
so the scoring step is the gene-level analogue of the paper's per-edge scoring.

One forward pass per cell type fills an `AttentionCache` (model specific:
`collect_scgpt`, `collect_geneformer` for Geneformer and scCello); everything after it (pooled means, Leiden strata, CSSI,
donor-level matrices in attention_dynamic_selection.py) only reads the cache,
so it is the same code for every model.

Every mean is over all the cells of the group (condition, stratum or donor),
a cell without the gene adding 0: a gene seen in a handful of cells cannot
outrank one seen in all of them.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import multiprocessing as mp
from typing import Callable, Dict, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from einops import rearrange

from . import _step7_common as common
from . import _step7_geneformer as gf
from .. import config as C
from ..utils.progress import progress

DEFAULT_N_NEIGHBORS = 15
DEFAULT_TARGET_K = 6
DEFAULT_RESOLUTIONS = tuple(np.geomspace(0.1, 2.5, 25))
# The number of Leiden clusters grows with the resolution, so once this many
# resolutions in a row give more than target_k clusters, the higher ones
# (much slower on large cell types) cannot come closer to target_k. 0 sweeps
# every resolution. On scGPT astrocytes of Dataset A (24k cells) it gives the
# same strata as the full sweep in about a third of the time
LEIDEN_PATIENCE = 3
# Leiden is single-threaded, so on cell types of at least
# LEIDEN_PARALLEL_MIN_CELLS cells the resolutions run in batches of
# LEIDEN_PROCESSES parallel processes. Each process takes ~40 s to start
# (imports), so smaller cell types run one resolution at a time. Every Leiden
# call is deterministic, so the strata are the same either way (scGPT
# astrocytes of Dataset A, 24k cells: 3.9 min instead of 7.7; the gain grows
# with the time per resolution, i.e. with the cell type)
LEIDEN_PROCESSES = 8
LEIDEN_PARALLEL_MIN_CELLS = 10000


# ---------------------------------------------------------------------------
# Attention cache: one forward pass, every gene token of every cell
# ---------------------------------------------------------------------------
@dataclass
class AttentionCache:
    """CLS attention of every real gene token (no CLS/pad/special) of every cell.

    Tokens are stored flat, cell after cell: the tokens of cell i are
    `gene_idx[cell_ptr[i]:cell_ptr[i + 1]]`, `gene_idx` indexes `genes`.
    """
    gene_idx: np.ndarray      # (n_tokens,) int32
    attn: np.ndarray          # (n_tokens,) float32
    cell_ptr: np.ndarray      # (n_cells + 1,) int64
    labels: np.ndarray        # (n_cells,) int, 0 = control, 1 = PD
    embeddings: np.ndarray    # (n_cells, d) float32, residual-stream CLS embedding
    genes: np.ndarray         # (n_genes,) gene symbols
    embedding_source: str
    # (n_cells,) True where a fine-tuned classifier predicts the true label;
    # None for zero-shot models (no classification head)
    correct: Optional[np.ndarray] = None
    _token_cells: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_cells(self) -> int:
        return int(self.labels.size)

    def save(self, path) -> None:
        """Per-cell token scores to one .npz (embeddings excluded: cell_embeddings.npy)."""
        np.savez(path, gene_idx=self.gene_idx, attn=self.attn, cell_ptr=self.cell_ptr, labels=self.labels,
                 genes=self.genes.astype(str), embedding_source=np.asarray(self.embedding_source),
                 correct=np.zeros(0, bool) if self.correct is None else self.correct)

    @classmethod
    def load(cls, path, embeddings: np.ndarray) -> "AttentionCache":
        with np.load(path, allow_pickle=False) as npz:
            correct = npz["correct"]
            return cls(gene_idx=npz["gene_idx"], attn=npz["attn"], cell_ptr=npz["cell_ptr"],
                       labels=npz["labels"], embeddings=embeddings, genes=npz["genes"],
                       embedding_source=str(npz["embedding_source"]),
                       correct=correct if correct.size else None)

    # Cell index of every token
    @property
    def token_cells(self) -> np.ndarray:
        if self._token_cells is None:
            self._token_cells = np.repeat(
                np.arange(self.n_cells, dtype=np.int32), np.diff(self.cell_ptr))
        return self._token_cells


class _CacheBuilder:
    """Accumulates forward-pass batches and builds an AttentionCache."""

    def __init__(self):
        self._gids, self._attn, self._lens = [], [], []
        self._labels, self._embs, self._preds = [], [], []

    # gids/attn: (B, L) arrays; keep: (B, L) True on the gene tokens to store;
    # preds: (B,) predicted labels of a fine-tuned classifier, if any
    def add(self, gids: np.ndarray, attn: np.ndarray, keep: np.ndarray,
            labels: np.ndarray, embeddings: np.ndarray, preds: Optional[np.ndarray] = None) -> None:
        self._gids.append(gids[keep].astype(np.int64))
        self._attn.append(attn[keep].astype(np.float32))
        self._lens.append(keep.sum(axis=1))
        self._labels.append(np.asarray(labels, dtype=int))
        self._embs.append(np.asarray(embeddings, dtype=np.float32))
        if preds is not None:
            self._preds.append(np.asarray(preds, dtype=int))

    # `order[p]` is the original index of the p-th cell added (cells batched
    # out of order, e.g. by length); the cache is put back in original order
    def build(self, id_to_gene: Callable[[np.ndarray], Sequence[str]],
              embedding_source: str, order: Optional[np.ndarray] = None) -> AttentionCache:
        gids = np.concatenate(self._gids)
        attn = np.concatenate(self._attn)
        lens = np.concatenate(self._lens)
        labels = np.concatenate(self._labels)
        embs = np.concatenate(self._embs)
        preds = np.concatenate(self._preds) if self._preds else None
        if order is not None:
            back = np.argsort(order)
            starts = np.concatenate([[0], np.cumsum(lens)])[:-1][back]
            lens = lens[back]
            new_starts = np.concatenate([[0], np.cumsum(lens)])[:-1]
            tokens = np.repeat(starts - new_starts, lens) + np.arange(lens.sum())
            gids, attn, labels, embs = gids[tokens], attn[tokens], labels[back], embs[back]
            preds = None if preds is None else preds[back]
        # Compact gene index: the cache only knows the genes it has seen
        uniq, inv = np.unique(gids, return_inverse=True)
        return AttentionCache(
            gene_idx=inv.astype(np.int32),
            attn=attn,
            cell_ptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            labels=labels,
            embeddings=embs,
            genes=np.asarray([str(g) for g in id_to_gene(uniq)]),
            embedding_source=embedding_source,
            correct=None if preds is None else preds == labels,
        )


def _forward_cls_attention_with_embedding(
    model: torch.nn.Module,
    batch_pt: Dict[str, torch.Tensor],
    vocab,
    n_head: int,
    device: torch.device,
    n_cls: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CLS attention, predictions, and residual-stream CLS embedding."""
    pad_id = vocab[C.PAD_TOKEN]
    gids = batch_pt["gene_ids"].to(device)
    vals = batch_pt["values"].to(device)
    mask = gids.eq(pad_id)

    capture = common._LayerInputCapture(
        model.transformer_encoder.layers[common.NUM_ATTN_LAYERS]
    )
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(
                gids,
                vals,
                src_key_padding_mask=mask,
                batch_labels=None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
            )
    finally:
        capture.remove()

    embs = capture.captured.float()
    self_attn = model.transformer_encoder.layers[common.NUM_ATTN_LAYERS].self_attn
    qkv = rearrange(
        self_attn.Wqkv(embs),
        "b s (three h d) -> b s three h d",
        three=3,
        h=n_head,
    )
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]
    q_cls = q[:, 0:1, :, :].permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 3, 1)
    cls_scores = (q_cls @ k_t).squeeze(2)
    cls_attn = common._cls_attn_rank_norm_head_avg(cls_scores, mask)

    cls_output = output["cls_output"].float()
    if n_cls == 1:
        logits = cls_output.squeeze(1)
        preds = (logits > 0).long()
    else:
        preds = cls_output.argmax(dim=1)

    # CLS token entering the final attention layer: residual-stream cell embedding.
    cell_embeddings = embs[:, 0, :]
    return cls_attn, preds, cell_embeddings


SCGPT_ATTN_BATCH_SIZE = 16


def collect_scgpt(model, pt: Dict[str, torch.Tensor], vocab, gene_names: dict, n_head: int,
                  device: torch.device, n_cls: int = 2, use_predictions: bool = False,
                  batch_size: int = SCGPT_ATTN_BATCH_SIZE, desc: str = "attention") -> AttentionCache:
    """One scGPT forward pass over the tokenised cells of one cell type.

    Genes are named by `gene_names` (token id -> current symbol, GBA1 rather
    than the vocabulary's GBA). With `use_predictions` (LoRA classifier,
    n_cls=1) the cache also records which cells the classifier gets right
    (`cache.correct`).
    """
    special_ids = np.array(sorted({vocab[t] for t in C.SPECIAL_TOKENS if t in vocab}))
    builder = _CacheBuilder()
    n_cells = pt["gene_ids"].shape[0]
    with progress(total=n_cells, desc=desc, unit="cell") as bar:
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch = {k: v[start:end] for k, v in pt.items()}
            cls_attn, preds, embeddings = _forward_cls_attention_with_embedding(
                model, batch, vocab, n_head, device, n_cls
            )
            gids = batch["gene_ids"].numpy()
            # Position 0 is the CLS itself; pads and special tokens are not genes
            keep = ~np.isin(gids, special_ids)
            keep[:, 0] = False
            builder.add(gids, cls_attn.cpu().numpy(), keep,
                        batch["condition_labels"].numpy(), embeddings.cpu().numpy(),
                        preds.cpu().numpy() if use_predictions else None)
            bar.update(end - start)
    return builder.build(
        lambda ids: [gene_names[int(i)] for i in ids],
        f"scGPT{' LoRA' if use_predictions else ''} residual-stream CLS embedding entering "
        f"transformer layer {common.NUM_ATTN_LAYERS}",
    )


def collect_geneformer(model, pt: Dict, token_dict: dict, gene_names: dict, device: torch.device,
                       emb_layer: int, model_label: str = "Geneformer",
                       batch_size: int = gf.ATTN_BATCH_SIZE,
                       use_predictions: bool = False, desc: str = "attention") -> AttentionCache:
    """One Geneformer (or scCello) forward pass over the tokenised cells of one cell type.

    The attention and the CSSI embedding are those of the block whose output
    is the cell embedding (`emb_layer`, see _step7_geneformer.attn_layer_index).
    Cells are batched by length so each batch carries little padding (cells
    have 165-4096 tokens); the cache comes back in the original cell order.
    With `use_predictions` (LoRA classifier) the cache also records which
    cells the classifier gets right (`cache.correct`).
    """
    # <pad>, <mask>, <cls> and <eos> (Geneformer only) are not genes
    special_ids = np.array(sorted(v for k, v in token_dict.items() if k.startswith("<")))
    pad_id = token_dict["<pad>"]
    layer_idx = gf.attn_layer_index(model, emb_layer)
    labels = pt["condition_labels"].numpy()
    order = np.argsort([len(s) for s in pt["input_ids"]], kind="stable")
    builder = _CacheBuilder()
    # Shortest cells first, so the rate falls as the batches get longer
    with progress(total=len(order), desc=desc, unit="cell") as bar:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids, attn = gf._pad_batch([pt["input_ids"][i] for i in idx], pad_id, device)
            cls_attn, preds, embeddings = gf._forward_cls_attention_with_embedding(
                model, ids, attn, layer_idx, use_predictions)
            gids = ids.cpu().numpy()
            keep = ~np.isin(gids, special_ids)
            keep[:, 0] = False
            builder.add(gids, cls_attn.cpu().numpy(), keep, labels[idx], embeddings.cpu().numpy(),
                        None if preds is None else preds.cpu().numpy())
            bar.update(len(idx))
    return builder.build(
        lambda ids: [gene_names[int(i)] for i in ids],
        f"{model_label} residual-stream CLS embedding entering encoder layer {layer_idx}",
        order=order,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def condition_means(cache: AttentionCache,
                    cell_mask: Optional[np.ndarray] = None) -> Dict[int, Dict[str, float]]:
    """Mean attention per gene per condition over the cells in `cell_mask`.

    The denominator is the number of cells of the condition, so a cell
    without the gene adds 0. Genes absent from a condition are left out of its
    dict (common._gene_scores treats them as 0).
    """
    n_genes = cache.genes.size
    token_cells = cache.token_cells
    means: Dict[int, Dict[str, float]] = {0: {}, 1: {}}
    for cond in (0, 1):
        cells = cache.labels == cond
        if cell_mask is not None:
            cells &= cell_mask
        n_used = int(cells.sum())
        if n_used == 0:
            continue
        tokens = cells[token_cells]
        gidx = cache.gene_idx[tokens]
        sums = np.bincount(gidx, weights=cache.attn[tokens], minlength=n_genes)
        present = np.bincount(gidx, minlength=n_genes) > 0
        means[cond] = dict(zip(cache.genes[present].tolist(), (sums[present] / n_used).tolist()))
    return means


# k-NN graph of the Leiden worker processes, set once per process
_LEIDEN_GRAPH = None


def _set_leiden_graph(graph) -> None:
    global _LEIDEN_GRAPH
    _LEIDEN_GRAPH = graph


def _leiden_labels(resolution: float, random_state: int, graph=None) -> np.ndarray:
    """Leiden cluster of every cell at one resolution (graph: the worker's if None)."""
    graph = _LEIDEN_GRAPH if graph is None else graph
    adata = ad.AnnData(X=np.zeros((graph.shape[0], 1), dtype=np.float32))
    sc.tl.leiden(adata, adjacency=graph, resolution=float(resolution), random_state=random_state,
                 key_added="_leiden", directed=False)
    return adata.obs["_leiden"].astype(int).to_numpy()


def _leiden_from_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    target_k: int = DEFAULT_TARGET_K,
    resolutions=DEFAULT_RESOLUTIONS,
    random_state: int = 42,
    embedding_source: str = "",
    desc: str = "Leiden",
    patience: int = LEIDEN_PATIENCE,
):
    """Build cosine k-NN graph and choose Leiden resolution closest to target_k.

    Resolutions are tried in increasing order, stopping early after
    `patience` consecutive resolutions above target_k clusters, in parallel
    batches on large cell types (LEIDEN_PROCESSES).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 3:
        return np.zeros(embeddings.shape[0], dtype=int), {
            "n_neighbors": min(n_neighbors, max(1, embeddings.shape[0] - 1)),
            "resolution": 0.0,
            "n_clusters": 1,
        }

    graph_adata = ad.AnnData(X=embeddings)
    nn = min(n_neighbors, embeddings.shape[0] - 1)
    sc.pp.neighbors(
        graph_adata,
        n_neighbors=nn,
        use_rep="X",
        metric="cosine",
        random_state=random_state,
    )
    graph = graph_adata.obsp["connectivities"]

    # Results are read in resolution order, so the early stop and the choice
    # below see exactly what a one-at-a-time sweep sees
    resolutions = sorted(resolutions)
    n_proc = LEIDEN_PROCESSES if embeddings.shape[0] >= LEIDEN_PARALLEL_MIN_CELLS else 1
    # spawn, not fork: the calling process may hold a CUDA context
    pool = (ProcessPoolExecutor(n_proc, mp_context=mp.get_context("spawn"),
                                initializer=_set_leiden_graph, initargs=(graph,)) if n_proc > 1 else None)
    candidates, sweep, above, stop = [], [], 0, False
    try:
        with progress(total=len(resolutions), desc=desc, unit="resolution") as bar:
            for start in range(0, len(resolutions), n_proc):
                batch = resolutions[start:start + n_proc]
                results = (pool.map(_leiden_labels, batch, [random_state] * len(batch)) if pool
                           else (_leiden_labels(r, random_state, graph) for r in batch))
                for resolution, labels in zip(batch, results):
                    n_clusters = int(np.unique(labels).size)
                    candidates.append((abs(n_clusters - target_k), -n_clusters, float(resolution), labels))
                    sweep.append([float(resolution), n_clusters])
                    bar.update(1)
                    above = above + 1 if n_clusters > target_k else 0
                    if patience and above >= patience:
                        stop = True
                        break
                if stop:
                    break
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    _, _, best_resolution, best_labels = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    meta = {
        "n_neighbors": nn,
        "resolution": best_resolution,
        "n_clusters": int(np.unique(best_labels).size),
        "target_k": target_k,
        "metric": "cosine",
        "embedding_source": embedding_source,
        # [resolution, number of clusters] of every resolution tried
        "resolution_sweep": sweep,
        "leiden_patience": patience,
    }
    return best_labels.astype(int), meta


def _signed(mean_pd: float, mean_ctrl: float) -> float:
    return mean_pd - mean_ctrl


def compute_attention_cssi(
    cache: AttentionCache,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    target_k: int = DEFAULT_TARGET_K,
    min_cells_per_condition: int = 20,
    random_state: int = 42,
    desc: str = "Leiden",
    cell_mask: Optional[np.ndarray] = None,
    precomputed_strata: Optional[Tuple[np.ndarray, dict]] = None,
):
    """Pooled means, CSSI-max gene scores and Leiden labels from one cache.

    Leiden clusters every cell (or `precomputed_strata` = (strata, metadata)
    of an earlier run is used); the scores only use the cells in `cell_mask`
    (all by default; the correctly classified ones for a fine-tuned
    classifier), and a stratum needs min_cells_per_condition of them per
    condition, so every mean it contributes rests on enough cells. A
    fine-tuned model's embedding groups cells by its prediction, so few of
    its strata hold enough correctly classified cells of both conditions.
    Returns (pooled_means, cssi_scores, gate, strata, metadata).
    """
    n_cells = cache.n_cells
    labels = cache.labels
    scored = np.ones(n_cells, dtype=bool) if cell_mask is None else np.asarray(cell_mask, dtype=bool)
    n_per_cond = {c: int(np.sum(scored & (labels == c))) for c in (0, 1)}
    if n_per_cond[0] == 0 or n_per_cond[1] == 0:
        gate = {
            "ok": False,
            "reason": f"ground-truth counts: control={n_per_cond[0]}, pd={n_per_cond[1]}",
            "mode": "labels",
        }
        return {}, {}, gate, np.zeros(n_cells, dtype=int), {}

    if precomputed_strata is not None:
        strata, cluster_meta = precomputed_strata[0], dict(precomputed_strata[1])
    else:
        strata, cluster_meta = _leiden_from_embeddings(
            cache.embeddings,
            n_neighbors=n_neighbors,
            target_k=target_k,
            random_state=random_state,
            embedding_source=cache.embedding_source,
            desc=desc,
        )

    pooled_means = condition_means(cache, cell_mask)
    cssi_scores = {}
    valid_strata = 0

    for stratum in np.unique(strata):
        in_stratum = (strata == stratum) & scored
        cells_control = int(np.sum(in_stratum & (labels == 0)))
        cells_pd = int(np.sum(in_stratum & (labels == 1)))
        if min(cells_control, cells_pd) < min_cells_per_condition:
            continue
        valid_strata += 1

        scores = common._gene_scores(condition_means(cache, in_stratum), _signed)
        for gene, score in scores.items():
            # CSSI-max is the maximum signed stratum-specific contrast.
            if gene not in cssi_scores or score > cssi_scores[gene]:
                cssi_scores[gene] = float(score)

    if valid_strata == 0:
        # No valid PD/control-balanced stratum: do not invent a contrast.
        cssi_scores = common._gene_scores(pooled_means, _signed)

    cluster_meta["valid_strata"] = valid_strata
    cluster_meta["n_cells"] = int(n_cells)
    if cell_mask is not None:
        cluster_meta["n_cells_scored"] = int(scored.sum())
    cluster_meta["n_embedding_dims"] = int(cache.embeddings.shape[1])
    return pooled_means, cssi_scores, {
        "ok": True,
        "reason": "",
        "mode": "labels",
    }, strata, cluster_meta


def save_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)
