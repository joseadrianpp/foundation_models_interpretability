"""
Reading the astrocytes of one dataset and writing the cell embeddings of a
model. Shared by the three embedding scripts, so it only uses packages that
the three environments have (anndata, numpy).
"""

import anndata as ad
import numpy as np

import config as C


# The three files of a dataset in one AnnData, with the set of each cell in obs["split"]
def load_astrocytes(dataset: str) -> ad.AnnData:
    parts = []
    for split in C.SPLITS:
        part = ad.read_h5ad(C.DATASET_DIRS[dataset] / f"astrocytes_{split}.h5ad")
        part.obs[C.SPLIT_COL] = split
        parts.append(part)
    adata = ad.concat(parts, merge="same")
    print(f"Dataset {dataset}: {adata.n_obs} astrocytes x {adata.n_vars} genes")
    return adata


# The genes a model reads: those with a name in its column of var
def model_genes(adata: ad.AnnData, gene_col: str) -> ad.AnnData:
    adata = adata[:, adata.var[gene_col] != ""].copy()
    print(f"Genes read by the model: {adata.n_vars}")
    return adata


# The embeddings go in X, one row per cell in the order of `adata`, with the
# cell metadata (donor, diagnosis, split...) so the MLP needs nothing else
def save_embeddings(adata: ad.AnnData, embeddings: np.ndarray, dataset: str, model: str) -> None:
    assert len(embeddings) == adata.n_obs, "one embedding per cell"
    out = ad.AnnData(X=np.asarray(embeddings, dtype=np.float32), obs=adata.obs.copy())
    path = C.embeddings_path(dataset, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(path)
    print(f"Embeddings {out.shape} saved in {path}")
