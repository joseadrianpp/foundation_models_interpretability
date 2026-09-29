"""Trimmed constants for the interpretability-only reproduction package.

Mirrors the subset of code/pipeline_scgpt/config.py actually read by the
copied step modules (steps/_step7_common.py, step7_explainability_*.py) and
their utils (utils/scgpt_model.py, utils/splits.py). The Geneformer constants
(steps/_step7_geneformer.py, utils/geneformer_model.py) mirror
code/pipeline_unified/config.py.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# scGPT brain pretrained model (used by the zero-shot attention task and by
# the architecture metadata for the LoRA IG task).
SCGPT_MODEL_DIR = DATA_DIR / "scGPT_brain"

# scGPT generates a universe of embeddings for all genes
# each gene has an embedding defined during the model's pre-training (MLM)
SCGPT_VOCAB = SCGPT_MODEL_DIR / "vocab.json"
SCGPT_WEIGHTS = SCGPT_MODEL_DIR / "best_model.pt"
SCGPT_ARGS = SCGPT_MODEL_DIR / "args.json"

# Head of the LoRA checkpoint: the Step 4 MLP it was warm-started from
# (pipeline_unified Step 4 best_params.json, same shape in Dataset A and B).
# The checkpoint stores only weights, so the head is rebuilt with this shape.
SCGPT_LORA_HEAD_CONFIG = {
    "input_dim": 512, "n_layers": 2, "hidden_size": 64, "dropout": 0.3,
    "activation": "gelu", "mlp_type": "constant", "n_classes": 1,
}

# Geneformer V2 104M pretrained model (used by the zero-shot attention task and
# as the backbone of the Geneformer LoRA). It is a plain BERT, so transformers
# loads it; the cells come already tokenized, so the geneformer package itself
# is not needed.
GENEFORMER_MODEL_DIR = DATA_DIR / "Geneformer_V2_104M"

# Geneformer vocabulary: Ensembl ID -> token id, plus <pad>, <mask>, <cls>, <eos>
GENEFORMER_TOKEN_DICT = GENEFORMER_MODEL_DIR / "token_dictionary_gc104M.pkl"

# Geneformer tokens are Ensembl IDs, this maps every token of our data to its
# gene symbol (one file inside each dataset folder, data/dataset_a or data/dataset_b)
GENEFORMER_GENE_NAMES = Path("mixed_split_geneformer") / "gene_names.csv"

# Head of the Geneformer LoRA checkpoint, again the Step 4 MLP it was
# warm-started from (same shape in Dataset A and B)
GENEFORMER_LORA_HEAD_CONFIG = {
    "input_dim": 768, "n_layers": 2, "hidden_size": 64, "dropout": 0.3,
    "activation": "gelu", "mlp_type": "constant", "n_classes": 1,
}

# Column names in the test h5ad obs
CELL_TYPE_COL = "cell_type"
# Binary diagnosis of each cell, the same column in Dataset A and Dataset B
LABEL_COL = "diagnosis"

# Label normalisation: free-text -> canonical -> integer
LABEL_NORMALISE = {
    "parkinson": "pd",
    "parkinson's": "pd",
    "pd": "pd",
    "control": "control",
    "unaffected control": "control",
}
LABEL_TO_INT = {"control": 0, "pd": 1}

# scGPT special tokens (must mirror the original training run)
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = -2
N_BINS = 51
MAX_SEQ_LEN = 1001  # n_top_genes (1000) + 1, as used for this run

# go3/gwas_similarity/gene_matrix postprocess steps expect the monorepo's own
# GO ontology and gene-annotation file symlinked here (data/ is gitignored):
#   data/go3_refs/go-basic.obo  -> ../../../pipeline_unified/data/ontologies/go-basic.obo
#   data/go3_refs/goa_human.gaf -> ../../../pipeline_unified/data/goa_human.gaf
# go3 itself is not on PyPI; it must already be importable in the environment
# (it is, inside scgpt_env).
