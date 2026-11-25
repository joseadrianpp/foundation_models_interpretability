#!/usr/bin/env python
"""
Zero-shot cell embeddings of scCello, with the authors' code.

Follows scCello's own scripts (sccello/script/run_data_transformation.py and
the cell type clustering and classification ones):

  1. ProcessSingleCellData.from_h5ad_adata: the same rank value encoding as
     Geneformer (counts over the cell's total x 10^4, divided by each gene's
     median, genes ranked from highest to lowest, <cls> in front).
  2. TranscriptomeTokenizer: keeps the first 2048 tokens.
  3. The encoder of their classification model (the one their linear probing
     trains) and the <cls> output of its last layer, with no projection or
     normalisation, as their zero-shot clustering reads it (pass_cell_cls=0,
     normalize=0).

scCello reads genes by Ensembl ID, the column ensembl_id_sccello of the data.

Run in sccello_env (classification/envs/sccello_env.txt):

    python classification/embed_sccello.py --dataset A
"""

import argparse
import sys

import anndata as ad
import numpy as np
import torch
from datasets import concatenate_datasets
from torch.utils.data import DataLoader

import config as C
from cells import load_astrocytes, model_genes, save_embeddings

sys.path.insert(0, str(C.SCCELLO_REPO))
# Their modules import each other in a circle that only resolves if data_loading
# is imported before the model, so this line has to stay the first one
from sccello.src.utils import data_loading
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification
from sccello.src.data.data_proc import ProcessSingleCellData
from sccello.src.data.tokenizer import TranscriptomeTokenizer
from sccello.src.model_prototype_contrastive import PrototypeContrastiveForSequenceClassification

# Their preprocessing of data that is not from CellxGene needs a dense matrix,
# so the cells go through it in blocks to keep the memory low. Each cell is
# encoded on its own, so the blocks do not change the result
BLOCK_SIZE = 2000


# Rank value encoding and truncation to 2048 tokens, with their functions
def tokenize(adata: ad.AnnData):
    var = adata.var[[]].assign(ensembl_ids=adata.var[C.SCCELLO_GENE_COL].values)
    blocks = []
    for start in range(0, adata.n_obs, BLOCK_SIZE):
        block = ad.AnnData(X=adata.X[start:start + BLOCK_SIZE].toarray(), var=var)
        blocks.append(ProcessSingleCellData.from_h5ad_adata(
            block, "astrocytes", is_cellxgene=False, require_raw=False))
    cells = concatenate_datasets(blocks)
    return TranscriptomeTokenizer(num_proc=4, truncate_length=C.SCCELLO_INPUT_SIZE).tokenize(cells)


# <cls> output of the last layer of the encoder, for every cell in order
def embed(tokens, device: torch.device) -> np.ndarray:
    # Their attention layers only have the "eager" implementation
    model = PrototypeContrastiveForSequenceClassification.from_pretrained(
        str(C.SCCELLO_MODEL_DIR), num_labels=2, attn_implementation="eager")
    model.to(device).eval()

    # Their collator pads each batch to its longest cell and builds the attention mask
    collator = DataCollatorForCellClassification(model_input_name=["input_ids"])
    loader = DataLoader(tokens.select_columns(["input_ids"]), batch_size=C.SCCELLO_BATCH_SIZE,
                        shuffle=False, collate_fn=collator)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            out = model.bert(input_ids=batch["input_ids"].to(device),
                             attention_mask=batch["attention_mask"].to(device),
                             output_hidden_states=False)
            embeddings.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
    return np.vstack(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", choices=sorted(C.DATASET_DIRS), required=True)
    args = parser.parse_args()

    adata = model_genes(load_astrocytes(args.dataset), C.SCCELLO_GENE_COL)
    tokens = tokenize(adata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_embeddings(adata, embed(tokens, device), args.dataset, "sccello")


if __name__ == "__main__":
    main()
