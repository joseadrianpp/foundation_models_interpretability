#!/usr/bin/env python
"""
Zero-shot cell embeddings of Geneformer V2 (104M), with the authors' code.

Follows Geneformer's tokenizing and embedding-extraction examples
(examples/tokenizing_scRNAseq_data.ipynb and examples/extract_and_plot_cell_embeddings.ipynb):

  1. TranscriptomeTokenizer: divides each gene's counts by the cell's total
     counts and by the gene's median over Geneformer's pretraining corpus,
     ranks the expressed genes from highest to lowest (rank value encoding),
     puts <cls> in front and <eos> at the end, and keeps the first 4096 tokens.
  2. EmbExtractor: the <cls> output of the second-to-last layer (emb_layer=-1),
     the layer the authors recommend for a general cell representation.

Geneformer reads genes by Ensembl ID, the column ensembl_id_geneformer of the data.

Run in geneformer_env (classification/envs/geneformer_env.txt plus the geneformer package):

    python classification/embed_geneformer.py --dataset A
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from geneformer import EmbExtractor, TranscriptomeTokenizer

import config as C
from cells import load_astrocytes, model_genes, save_embeddings


# Tokenize the cells with Geneformer's tokenizer. It reads h5ad files from a
# folder and needs the Ensembl ID of each gene (var["ensembl_id"]) and the
# total counts of each cell (obs["n_counts"]). The barcode goes along with each
# cell, because the embedding extractor returns the cells in another order
def tokenize(adata, work_dir: Path) -> Path:
    adata.var["ensembl_id"] = adata.var[C.GENEFORMER_GENE_COL]
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["barcode"] = adata.obs_names.astype(str)
    adata.write_h5ad(work_dir / "astrocytes.h5ad")

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={"barcode": "barcode"},
        model_input_size=C.GENEFORMER_INPUT_SIZE, special_token=True, model_version="V2", nproc=4,
    )
    tokenizer.tokenize_data(work_dir, work_dir, "astrocytes", file_format="h5ad")
    return work_dir / "astrocytes.dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", choices=sorted(C.DATASET_DIRS), required=True)
    args = parser.parse_args()

    adata = model_genes(load_astrocytes(args.dataset), C.GENEFORMER_GENE_COL)
    with tempfile.TemporaryDirectory() as tmp:
        tokens = tokenize(adata, Path(tmp))

        # max_ncells=None embeds every cell (by default it samples 1000 of them)
        extractor = EmbExtractor(
            model_type="Pretrained", num_classes=0, emb_mode="cls", emb_layer=-1,
            max_ncells=None, emb_label=["barcode"], forward_batch_size=C.GENEFORMER_BATCH_SIZE,
            model_version="V2", nproc=4,
        )
        embs = extractor.extract_embs(C.GENEFORMER_MODEL_DIR, tokens, tmp, "embeddings")

    # One row per cell with the embedding columns and its barcode, longest cells
    # first; back to the order of the data. A cell with no gene of the
    # vocabulary would be dropped by the tokenizer and stay missing here
    embs = embs.set_index("barcode").reindex(adata.obs_names)
    assert not embs.isna().any().any(), "some cells were not embedded"
    save_embeddings(adata, embs.to_numpy(), args.dataset, "geneformer")


if __name__ == "__main__":
    main()
