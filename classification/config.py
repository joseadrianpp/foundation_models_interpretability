"""
Paths and constants of the PD vs control classification of astrocytes from
frozen cell embeddings (scGPT, Geneformer and scCello + MLP head).

Each model's script runs in that model's own environment, so this file only
uses the standard library and every script can import it.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# The astrocytes of each dataset come as three files with the cells of
# different donors (train, validation and test), so no donor is in two of them
DATASET_DIRS = {"A": DATA_DIR / "dataset_a", "B": DATA_DIR / "dataset_b"}
SPLITS = ("train", "val", "test")
MODELS = ("scgpt", "geneformer", "sccello")

# Columns of the cell metadata (obs) of every h5ad
DONOR_COL = "donor"
LABEL_COL = "diagnosis"
PD_LABEL = "PD"
SPLIT_COL = "split"

# Each model reads the genes its vocabulary knows, under its own name. These
# var columns hold that name ("" if the model does not read the gene)
SCGPT_GENE_COL = "scgpt_gene"
GENEFORMER_GENE_COL = "ensembl_id_geneformer"
SCCELLO_GENE_COL = "ensembl_id_sccello"

# Zero-shot pretrained models, as their authors published them
SCGPT_MODEL_DIR = DATA_DIR / "scGPT_brain"
GENEFORMER_MODEL_DIR = DATA_DIR / "Geneformer-V2-104M"
SCCELLO_MODEL_DIR = DATA_DIR / "scCello-zeroshot"

# scCello's code is not an installable package, its own scripts import it
# from the root of its repository (a copy of it comes with the data)
SCCELLO_REPO = DATA_DIR / "githubs" / "scCello"

# Maximum input length of each model, the one it was pretrained with
SCGPT_MAX_LENGTH = 1200
GENEFORMER_INPUT_SIZE = 4096
SCCELLO_INPUT_SIZE = 2048

# Batch sizes for the forward pass (24 GB GPU). Geneformer has the longest
# cells, so it gets the smallest batch
SCGPT_BATCH_SIZE = 64
GENEFORMER_BATCH_SIZE = 16
SCCELLO_BATCH_SIZE = 32

SEED = 42

# MLP head on the frozen embeddings: the configuration of pipeline_unified's
# Step 4 (the head the LoRA fine-tuning starts from)
MLP_PARAMS = {
    "batch_size": 32, "lr": 1e-3, "dropout": 0.3, "weight_decay": 1e-3,
    "n_layers": 2, "hidden_size": 64, "activation": "gelu", "mlp_type": "constant",
}
MLP_EPOCHS = 100
# Stop when the validation balanced accuracy has not improved by more than
# MLP_MIN_DELTA in MLP_PATIENCE epochs, and keep the epoch with the lowest
# validation loss
MLP_PATIENCE = 5
MLP_MIN_DELTA = 0.01


# Where each model writes the embeddings of one dataset, and the MLP its results
def embeddings_path(dataset: str, model: str) -> Path:
    return RESULTS_DIR / f"dataset_{dataset.lower()}" / model / "embeddings.h5ad"


def mlp_dir(dataset: str, model: str) -> Path:
    return RESULTS_DIR / f"dataset_{dataset.lower()}" / model / "mlp"
