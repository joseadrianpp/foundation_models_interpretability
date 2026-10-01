"""Trimmed constants for the interpretability-only reproduction package.

Mirrors the subset of code/pipeline_scgpt/config.py actually read by the
copied step modules (steps/_step7_common.py, step7_explainability_*.py) and
their utils (utils/scgpt_model.py, utils/splits.py).
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

# Column names in the test h5ad obs
CELL_TYPE_COL = "cell_type"
LABEL_COL = "Brain_Bank_Path_Dx"  # raw diagnosis label

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
# scGPT's binning breaks ties with np.random, so tokenisation is seeded to make
# attention/IG scores independent of cell-type processing order.
BINNING_SEED = 42
MAX_SEQ_LEN = 1001  # n_top_genes (1000) + 1, as used for this run

# go3/gwas_similarity/gene_matrix postprocess steps expect the monorepo's own
# GO ontology and gene-annotation file symlinked here (data/ is gitignored):
#   data/go3_refs/go-basic.obo  -> ../../../../GO3/go-basic.obo
#   data/go3_refs/goa_human.gaf -> ../../../../data/goa_human.gaf
# go3 itself is not on PyPI; it must already be importable in the environment
# (it is, inside scgpt_env).
