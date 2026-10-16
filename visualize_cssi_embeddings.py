#!/usr/bin/env python
"""Visualize the cell embeddings and Leiden strata CSSI used, per score and cell type.

Every model task with CSSI saves its residual-stream CLS embeddings and Leiden
labels under results/dataset_<x>/<scores>/<cell type>/ (scores:
attention_zero_shot_<model>, attention_lora_<model>, ig_lora_<model>). This
script runs no model: it loads them and draws, per cell type, PCA / UMAP /
t-SNE colored by stratum, UMAP colored by condition, the cosine k-NN graph and
the cells of each stratum per condition (valid: >= 20 of each, as CSSI).

UMAP and t-SNE use at most --max-cells random cells; the strata are always
the saved ones. Figures go to
results/dataset_<x>/plots/cssi_embeddings/<scores>/<cell type>.png.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.manifold import TSNE

from pipeline import config as C
from pipeline.postprocess.plots import INK_2, MUTED, _style
from pipeline.steps._step7_common import _slug_cell_type
from pipeline.utils.splits import labels_to_int

HERE = Path(__file__).resolve().parent
CELL_TYPES = ["astrocytes", "da_neurons", "microglia", "oligodendrocytes"]
SCORES = [f"{score}_{model}" for model in ("scgpt", "geneformer", "sccello")
          for score in ("attention_zero_shot", "attention_lora", "ig_lora")]
# Every model sees the same cells in the same order; the labels come from the scGPT h5ad obs
SCGPT_H5ADS = [Path("mixed_split_scgpt") / f"Mixed_allgenes_pc_noXY_noMT_mixed_{split}.h5ad"
               for split in ("train", "val", "test")]
MIN_CELLS_PER_CONDITION = 20

# Categorical slots in fixed order (validated palette); strata past the 8th are gray
STRATUM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CONDITION_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CONDITION_NAMES = {0: "control", 1: "PD"}


def _stratum_colors(strata: np.ndarray) -> list:
    return [STRATUM_COLORS[s] if s < len(STRATUM_COLORS) else MUTED for s in strata]


def _scatter(ax, coords, colors, title):
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=4, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def _knn(ax, coords, connectivities, colors, max_nodes, rng, title):
    n = len(coords)
    keep = np.arange(n) if n <= max_nodes else np.sort(rng.choice(n, max_nodes, replace=False))
    sub = connectivities[keep][:, keep].tocoo()
    mask = sub.row < sub.col
    segments = [[coords[keep[i]], coords[keep[j]]] for i, j in zip(sub.row[mask], sub.col[mask])]
    if segments:
        ax.add_collection(LineCollection(segments, linewidths=0.25, alpha=0.15, colors=MUTED))
    ax.scatter(coords[keep, 0], coords[keep, 1], c=[colors[i] for i in keep], s=7, alpha=0.9, linewidths=0)
    ax.set_title(f"{title} ({len(keep):,} nodes shown)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def _composition(ax, strata, labels):
    """Cells per stratum and condition, every cell; valid strata marked."""
    ids = np.unique(strata)
    counts = {c: np.array([np.sum((strata == s) & (labels == c)) for s in ids]) for c in (0, 1)}
    y = np.arange(len(ids))
    ax.barh(y, counts[0], color=CONDITION_COLORS[0], height=0.7, label=CONDITION_NAMES[0])
    ax.barh(y, counts[1], left=counts[0], color=CONDITION_COLORS[1], height=0.7, label=CONDITION_NAMES[1])
    total = counts[0] + counts[1]
    for i in range(len(ids)):
        valid = min(counts[0][i], counts[1][i]) >= MIN_CELLS_PER_CONDITION
        ax.text(total[i], y[i], f"  {counts[0][i]:,} / {counts[1][i]:,}" + ("  valid" if valid else ""),
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"stratum {s}" for s in ids], fontsize=9)
    for tick, s in zip(ax.get_yticklabels(), ids):
        tick.set_color(STRATUM_COLORS[s] if s < len(STRATUM_COLORS) else MUTED)
    ax.invert_yaxis()
    ax.set_xlim(0, total.max() * 1.45)
    ax.set_xlabel("Cells (control / PD)")
    ax.set_title("Cells per stratum and condition", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)


def _visualize_one(scores_dir: Path, labels, out_path: Path, n_neighbors: int, n_pcs: int,
                   graph_max_nodes: int, max_cells: int, seed: int) -> str:
    emb_path, leiden_path = scores_dir / "cell_embeddings.npy", scores_dir / "leiden_clusters.csv"
    if not emb_path.exists() or not leiden_path.exists():
        return f"  skipped {scores_dir}: missing {emb_path.name} or {leiden_path.name}"
    embeddings = np.load(emb_path).astype(np.float32)
    strata = pd.read_csv(leiden_path)["leiden"].to_numpy()
    if len(strata) != len(embeddings):
        raise ValueError(f"Embedding/Leiden length mismatch in {scores_dir}")
    if labels is not None and len(labels) != len(strata):
        labels = None  # a --max-cells-per-type run: the cells are a subset
    meta_path = scores_dir / "cssi_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_cells, n_dims = embeddings.shape
    if n_cells < 5:
        return f"  skipped {scores_dir}: only {n_cells} cells"

    # Plot coordinates from a random subset; strata and composition use every cell
    rng = np.random.default_rng(seed)
    keep = np.arange(n_cells) if n_cells <= max_cells else np.sort(rng.choice(n_cells, max_cells, replace=False))
    a = ad.AnnData(X=embeddings[keep])
    sc.pp.pca(a, n_comps=max(2, min(n_pcs, n_dims - 1, len(keep) - 1)), svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(n_neighbors, len(keep) - 1), use_rep="X", metric="cosine", random_state=seed)
    sc.tl.umap(a, random_state=seed)
    perplexity = min(30.0, max(2.0, (len(keep) - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(a.obsm["X_pca"])
    colors = _stratum_colors(strata[keep])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    _scatter(axes[0, 0], a.obsm["X_pca"][:, :2], colors, "PCA · Leiden stratum")
    _scatter(axes[0, 1], a.obsm["X_umap"], colors, "UMAP · Leiden stratum")
    _scatter(axes[0, 2], tsne, colors, "t-SNE · Leiden stratum")
    if labels is not None:
        _scatter(axes[1, 0], a.obsm["X_umap"], [CONDITION_COLORS[c] for c in labels[keep]], "UMAP · condition")
        handles = [plt.Line2D([], [], ls="", marker="o", color=CONDITION_COLORS[c], label=CONDITION_NAMES[c])
                   for c in (0, 1)]
        axes[1, 0].legend(handles=handles, fontsize=8, frameon=False, loc="best")
        _composition(axes[1, 2], strata, labels)
    else:
        for ax in (axes[1, 0], axes[1, 2]):
            ax.text(0.5, 0.5, "condition labels unavailable", ha="center", va="center", color=MUTED,
                    transform=ax.transAxes)
            ax.set_axis_off()
    _knn(axes[1, 1], a.obsm["X_umap"], a.obsp["connectivities"], colors, graph_max_nodes, rng,
         "UMAP · cosine k-NN")
    strata_handles = [plt.Rectangle((0, 0), 1, 1, color=STRATUM_COLORS[s] if s < len(STRATUM_COLORS) else MUTED,
                                    label=f"stratum {s}") for s in np.unique(strata)]
    fig.legend(handles=strata_handles, loc="lower center", ncol=len(strata_handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    shown = f"{len(keep):,} of {n_cells:,} cells shown" if len(keep) < n_cells else f"{n_cells:,} cells"
    run_info = (f" · resolution {meta['resolution']:.3g} · {meta.get('valid_strata', '?')} valid strata in the run"
                if "resolution" in meta else "")
    fig.suptitle(f"{scores_dir.parent.name} · {scores_dir.name} · {shown} · {n_dims}-D embedding · "
                 f"{len(np.unique(strata))} Leiden strata{run_info}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"  saved {out_path}"


def _condition_labels(data_dir: Path) -> dict:
    """{cell-type slug: 0/1 label of every cell}, in the order the model tasks use."""
    obs = pd.concat([ad.read_h5ad(data_dir / p, backed="r").obs[[C.CELL_TYPE_COL, C.LABEL_COL]]
                     for p in SCGPT_H5ADS])
    slugs = np.array([_slug_cell_type(ct) for ct in obs[C.CELL_TYPE_COL].astype(str)])
    labels = labels_to_int(obs[C.LABEL_COL].to_numpy())
    return {slug: labels[slugs == slug] for slug in np.unique(slugs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    p.add_argument("--dataset", choices=["A", "B", "A+B"], default="A")
    p.add_argument("--scores", default=",".join(SCORES),
                   help="Comma-separated model task folders (default: all nine with CSSI).")
    p.add_argument("--cell-types", default=",".join(CELL_TYPES))
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--graph-max-nodes", type=int, default=2000)
    p.add_argument("--max-cells", type=int, default=20000, help="Random cells drawn in UMAP/t-SNE.")
    p.add_argument("--workers", type=int, default=8, help="Figures made in parallel.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results_dir = args.results_dir / f"dataset_{args.dataset.lower()}"
    # A+B pools the cells of A and B, in that order (run_tasks.DATASET_DIRS)
    per_dataset = [_condition_labels(HERE / "data" / f"dataset_{d.lower()}") for d in args.dataset.split("+")]
    labels = {slug: np.concatenate([lab[slug] for lab in per_dataset]) for slug in per_dataset[0]}
    cell_types = [x.strip() for x in args.cell_types.split(",") if x.strip()]
    jobs = [(results_dir / scores / ct, labels.get(ct),
             results_dir / "plots" / "cssi_embeddings" / scores / f"{ct}.png")
            for scores in [x.strip() for x in args.scores.split(",") if x.strip()] for ct in cell_types]
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(_visualize_one, d, l, o, args.n_neighbors, args.n_pcs, args.graph_max_nodes,
                             args.max_cells, args.seed) for d, l, o in jobs]
        for f in futures:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
