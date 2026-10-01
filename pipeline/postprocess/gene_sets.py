"""
Gene sets fed to the post-hoc analyses, per (method, selection, cell type).

Methods:
  * attention_zero_shot        - pooled CLS attention, PD - control (baseline)
  * attention_cssi             - CSSI-max attention over Leiden strata (method 2.1)
  * attention_residualized_ols - OLS residual of the pooled attention (method 2.3)
  * ig_lora                    - Integrated Gradients on the LoRA model

Selections:
  * top30 - static top-30 ranking
  * elbow - Kneedle elbow on the descending score curve (variable size)

IG only writes a top-30, so it has no elbow gene set.

top30 outputs keep the original locations (`results/<task>/...`,
`results/plots/<name>.png`); every other selection gets its own subfolder
(`results/<task>/<selection>/...`) and plot suffix (`<name>_<selection>.png`).
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

CELL_TYPES = ["astrocytes", "da_neurons", "microglia", "oligodendrocytes"]
SELECTIONS = ["top30", "elbow"]

# Gene-list CSV (columns rank, gene, score) relative to results_dir
GENE_LIST_PATHS = {
    "attention_zero_shot": {
        "top30": "attention_zero_shot/{ct}/top30_genes.csv",
        "elbow": "attention_zero_shot/{ct}/csv/elbow.csv",
    },
    "attention_cssi": {
        "top30": "attention_zero_shot/{ct}/cssi/csv/top30.csv",
        "elbow": "attention_zero_shot/{ct}/cssi/csv/elbow.csv",
    },
    "attention_residualized_ols": {
        "top30": "attention_residualized_ols/{ct}/csv/top30.csv",
        "elbow": "attention_residualized_ols/{ct}/csv/elbow.csv",
    },
    "ig_lora": {
        "top30": "ig_lora/{ct}/top30_genes.csv",
    },
}
METHODS = list(GENE_LIST_PATHS)


def methods_for(selection: str) -> List[str]:
    return [m for m in METHODS if selection in GENE_LIST_PATHS[m]]


def gene_list_path(results_dir: Path, method: str, selection: str, cell_type: str) -> Path:
    return results_dir / GENE_LIST_PATHS[method][selection].format(ct=cell_type)


# All genes of the list, highest score first (elbow lists are not truncated)
def load_genes(results_dir: Path, method: str, selection: str, cell_type: str) -> Optional[list]:
    path = gene_list_path(results_dir, method, selection, cell_type)
    if not path.exists():
        print(f"  MISSING {path}")
        return None
    df = pd.read_csv(path).sort_values("score", ascending=False)
    genes = df["gene"].astype(str).tolist()
    return genes or None


def output_dir(results_dir: Path, task: str, selection: str) -> Path:
    base = results_dir / task
    return base if selection == "top30" else base / selection


def plot_path(results_dir: Path, name: str, selection: str) -> Path:
    suffix = "" if selection == "top30" else f"_{selection}"
    return results_dir / "plots" / f"{name}{suffix}.png"
