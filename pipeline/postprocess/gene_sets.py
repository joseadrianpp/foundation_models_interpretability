"""
Gene sets fed to the post-hoc analyses, per (method, selection, cell type).

A method is one gene ranking of one model, `<ranking>_<model>`:
  * attention_zero_shot_<model>        - pooled CLS attention, PD - control (baseline)
  * attention_cssi_<model>             - CSSI-max attention over Leiden strata (method 2.1)
  * attention_residualized_ols_<model> - OLS residual of the pooled attention (method 2.3)
for model in scgpt / geneformer, plus the fine-tuned (LoRA) variants:
  * ig_lora_scgpt                      - Integrated Gradients on the LoRA scGPT model
  * attention_lora_geneformer          - CLS attention of the LoRA Geneformer model

Selections:
  * top30   - static top-30 ranking
  * z-score - genes with score > mean + 2.5 * std (variable size)
  * elbow   - Kneedle elbow on the descending score curve (variable size)

The LoRA variants only write a top-30, so they have no z-score/elbow gene set.
METHOD_META gives each method's (ranking, model, model form, label), the
columns that identify an experiment.

All paths are relative to the per-dataset results folder (results/dataset_a,
results/dataset_b). top30 outputs keep the original locations
(`<task>/...`, `plots/<name>.png`); every other selection gets its own
subfolder (`<task>/<selection>/...`) and plot suffix (`<name>_<selection>.png`).
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

CELL_TYPES = ["astrocytes", "da_neurons", "microglia", "oligodendrocytes"]
SELECTIONS = ["top30", "z-score", "elbow"]
MODELS = ["scgpt", "geneformer"]

# Per method: gene-list CSV per selection (columns rank, gene, score), the
# all-gene score CSV and the dynamic-selection summary, relative to results_dir
GENE_LIST_PATHS = {}
ALL_SCORE_PATHS = {}
SUMMARY_PATHS = {}
METHOD_META = {}


def _register(method: str, ranking: str, model: str, form: str, label: str, lists: dict,
              all_scores: Optional[str] = None, summary: Optional[str] = None) -> None:
    GENE_LIST_PATHS[method] = lists
    METHOD_META[method] = {"ranking": ranking, "model": model, "model_form": form, "label": label}
    if all_scores:
        ALL_SCORE_PATHS[method] = all_scores
    if summary:
        SUMMARY_PATHS[method] = summary


# Dynamic selections live in `<sel_dir>/<selection>.csv`, next to selection_summary.json
def _dynamic_lists(top30: str, sel_dir: str) -> dict:
    return {"top30": top30, **{s: f"{sel_dir}/{s}.csv" for s in SELECTIONS if s != "top30"}}


for _model in MODELS:
    _pooled = f"attention_zero_shot_{_model}/{{ct}}"
    _ols = f"attention_residualized_ols_{_model}/{{ct}}"
    _register(f"attention_zero_shot_{_model}", "attention", _model, "zero-shot", "zero-shot attention (pooled)",
              _dynamic_lists(f"{_pooled}/top30_genes.csv", f"{_pooled}/csv"),
              f"{_pooled}/csv/all_genes.csv", f"{_pooled}/csv/selection_summary.json")
    _register(f"attention_cssi_{_model}", "cssi", _model, "zero-shot", "CSSI-max attention (2.1)",
              _dynamic_lists(f"{_pooled}/cssi/csv/top30.csv", f"{_pooled}/cssi/csv"),
              f"{_pooled}/cssi_genes.csv", f"{_pooled}/cssi/csv/selection_summary.json")
    _register(f"attention_residualized_ols_{_model}", "ols", _model, "zero-shot", "OLS-residual attention (2.3)",
              _dynamic_lists(f"{_ols}/csv/top30.csv", f"{_ols}/csv"),
              f"{_ols}/csv/ols.csv", f"{_ols}/csv/selection_summary.json")
    if _model == "scgpt":
        _register("ig_lora_scgpt", "integrated_gradients", "scgpt", "lora", "LoRA (Integrated Gradients)",
                  {"top30": "ig_lora_scgpt/{ct}/top30_genes.csv"})
    else:
        _register("attention_lora_geneformer", "attention", "geneformer", "lora", "LoRA (attention)",
                  {"top30": "attention_lora_geneformer/{ct}/top30_genes.csv"})

METHODS = list(GENE_LIST_PATHS)


def methods_for(selection: str, model: Optional[str] = None) -> List[str]:
    return [m for m in METHODS if selection in GENE_LIST_PATHS[m]
            and (model is None or METHOD_META[m]["model"] == model)]


def model_of(method: str) -> str:
    return METHOD_META[method]["model"]


def gene_list_path(results_dir: Path, method: str, selection: str, cell_type: str) -> Path:
    return results_dir / GENE_LIST_PATHS[method][selection].format(ct=cell_type)


# All genes of the list, highest score first (dynamic lists are not truncated)
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
