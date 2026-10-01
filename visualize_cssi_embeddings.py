#!/usr/bin/env python
"""Visualize the exact model embeddings and Leiden/k-NN graph used by CSSI.

The attention run saves residual-stream CLS embeddings and Leiden labels under
results/dataset_*/attention_zero_shot_scgpt/<cell_type>/. This script performs no model
inference: it loads those artifacts, recomputes PCA/UMAP/t-SNE for plotting,
and saves one figure per cell type to results/plots/<cell_type>/.
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
CELL_TYPES = ["astrocytes", "da_neurons", "microglia", "oligodendrocytes"]


def _plot_embedding(ax, coords, labels, title):
    labels = np.asarray(labels).astype(str)
    _, codes = np.unique(labels, return_inverse=True)
    ax.scatter(coords[:, 0], coords[:, 1], c=codes, cmap="tab20", s=6, alpha=0.75, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_knn(ax, coords, connectivities, labels, max_nodes, seed, title):
    n = len(coords)
    rng = np.random.default_rng(seed)
    keep = np.arange(n) if n <= max_nodes else np.sort(rng.choice(n, max_nodes, replace=False))
    sub = connectivities[keep][:, keep].tocoo()
    mask = sub.row < sub.col
    segments = [[coords[keep[i]], coords[keep[j]]] for i, j in zip(sub.row[mask], sub.col[mask])]
    if segments:
        ax.add_collection(LineCollection(segments, linewidths=0.25, alpha=0.12))
    labels = np.asarray(labels).astype(str)
    _, codes = np.unique(labels, return_inverse=True)
    ax.scatter(coords[keep, 0], coords[keep, 1], c=codes[keep], cmap="tab20", s=8, alpha=0.9, linewidths=0)
    ax.set_title(f"{title} ({len(keep):,} nodes shown)")
    ax.set_xticks([]); ax.set_yticks([])


# Builds and saves the 2x2 PCA/UMAP/t-SNE/k-NN figure for a single cell
# type's embeddings + Leiden labels. Pulled out of visualize_cell_type so
# that function is free to be the per-project iterator (below).
def _visualize_one(attention_dir: Path, out_path: Path, n_neighbors: int, n_pcs: int, graph_max_nodes: int, seed: int):
    emb_path = attention_dir / "cell_embeddings.npy"
    leiden_path = attention_dir / "leiden_clusters.csv"
    if not emb_path.exists() or not leiden_path.exists():
        print(f"  skipped: missing {emb_path.name} or {leiden_path.name}")
        return

    embeddings = np.load(emb_path).astype(np.float32)
    labels = pd.read_csv(leiden_path)["leiden"].to_numpy()
    if len(labels) != len(embeddings):
        raise ValueError(f"Embedding/Leiden length mismatch in {attention_dir}")

    n_cells, n_dims = embeddings.shape
    if n_cells < 5:
        print(f"  skipped: only {n_cells} cells")
        return

    a = ad.AnnData(X=embeddings)
    n_pcs_eff = min(n_pcs, n_dims - 1, n_cells - 1)
    sc.pp.pca(a, n_comps=max(2, n_pcs_eff), svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(n_neighbors, n_cells - 1), use_rep="X", metric="cosine", random_state=seed)
    sc.tl.umap(a, random_state=seed)

    perplexity = min(30.0, max(2.0, (n_cells - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(a.obsm["X_pca"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _plot_embedding(axes[0, 0], a.obsm["X_pca"][:, :2], labels, "PCA · Leiden")
    _plot_embedding(axes[0, 1], a.obsm["X_umap"], labels, "UMAP · Leiden")
    _plot_embedding(axes[1, 0], tsne, labels, "t-SNE · Leiden")
    _plot_knn(axes[1, 1], a.obsm["X_umap"], a.obsp["connectivities"], labels, graph_max_nodes, seed, "UMAP · cosine k-NN + Leiden")

    slug = attention_dir.name
    fig.suptitle(f"{slug} · {n_cells:,} cells · {n_dims:,}-D model embeddings · {len(np.unique(labels))} Leiden clusters", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# Iterates every results/dataset_*/attention_zero_shot_scgpt/<cell_type>/
# subfolder and writes each cell type's figure to its own
# results/dataset_*/plots/<cell_type>/ subfolder, mirroring the attention layout.
def visualize_cell_type(results_dir: Path, cell_types, n_neighbors: int, n_pcs: int, graph_max_nodes: int, seed: int):
    for cell_type in cell_types:
        attention_dir = results_dir / "attention_zero_shot_scgpt" / cell_type
        out_path = results_dir / "plots" / cell_type / "embedding_leiden.png"
        print(f"\n# visualize | {cell_type} #")
        _visualize_one(attention_dir, out_path, n_neighbors, n_pcs, graph_max_nodes, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    p.add_argument("--cell-types", default=",".join(CELL_TYPES))
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--graph-max-nodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cell_types = [x.strip() for x in args.cell_types.split(",") if x.strip()]
    visualize_cell_type(
        args.results_dir / f"dataset_{args.dataset.lower()}",
        cell_types,
        args.n_neighbors,
        args.n_pcs,
        args.graph_max_nodes,
        args.seed,
    )


if __name__ == "__main__":
    main()
