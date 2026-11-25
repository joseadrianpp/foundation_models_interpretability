#!/usr/bin/env python
"""
Zero-shot cell embeddings of scGPT (brain model), with the authors' code.

Follows scGPT's zero-shot reference mapping tutorial
(tutorials/zero-shot/Tutorial_ZeroShot_Reference_Mapping.ipynb):

  1. Keep the genes of scGPT's vocabulary, as in its fine-tuning tutorials.
  2. Normalise every cell to 10^4 counts and log1p (scanpy).
  3. scgpt.tasks.embed_data: bins the expressed genes of each cell into 51
     bins, keeps up to 1200 of them (a random subset of a longer cell), puts
     <cls> in front and returns the <cls> output of the last layer,
     L2-normalised.

Run in scgpt_env (the repo's requirements.txt, with flash-attn):

    python classification/embed_scgpt.py --dataset A
"""

import argparse

import numpy as np
import scanpy as sc
import scgpt as scg
import torch

import config as C
from cells import load_astrocytes, model_genes, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", choices=sorted(C.DATASET_DIRS), required=True)
    args = parser.parse_args()

    adata = model_genes(load_astrocytes(args.dataset), C.SCGPT_GENE_COL)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # The binning breaks ties at random and a cell with more than 1199 expressed
    # genes keeps a random subset of them, both with numpy's global generator,
    # so it is seeded to get the same embeddings every run
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    adata = scg.tasks.embed_data(
        adata, C.SCGPT_MODEL_DIR, gene_col=C.SCGPT_GENE_COL,
        max_length=C.SCGPT_MAX_LENGTH, batch_size=C.SCGPT_BATCH_SIZE,
    )
    save_embeddings(adata, adata.obsm["X_scGPT"], args.dataset, "scgpt")


if __name__ == "__main__":
    main()
