"""Rows of the experiment table and the settings shared by its modules.

One row per gene list registered in postprocess/gene_sets.py, for every
results/dataset_<x> folder present: the key columns (KEY) identify the
experiment, the other columns tell the modules where its inputs are.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from ..postprocess import gene_sets as gs

# Columns that identify an experiment, in table order. Score is the signal a
# ranking is computed from: a LoRA model has both an attention and an IG
# CSSI / residuals ranking (zero-shot rows are all attention)
KEY = ["Ranking", "Score", "TopN", "Model", "ModelForm", "Dataset", "CellType"]

# Table labels of the gene_sets.py names
RANKING_LABELS = {"attention": "attention", "cssi": "CSSI", "ols": "residuals",
                  "integrated_gradients": "IG"}
SCORE_LABELS = {"attention": "attention", "ig": "IG"}
TOPN_LABELS = {"top30": "top30", "z-score": "zscore", "elbow": "elbow"}
MODEL_LABELS = {"scgpt": "scGPT", "geneformer": "Geneformer", "sccello": "scCello"}
MODEL_FORM_LABELS = {"zero-shot": "zero-shot", "lora": "LoRA"}
CELL_TYPE_LABELS = {"astrocytes": "astrocyte", "da_neurons": "neuron", "microglia": "microglia",
                    "oligodendrocytes": "oligodendrocyte"}

DEFAULT_MODEL_FORMS = tuple(MODEL_FORM_LABELS)


@dataclass
class Settings:
    """Options of a table build, read by the modules that need them."""
    results_root: Path
    # Similarity columns: the first distance gives Sim<ref>_<ontology>, the
    # others Sim<ref>_<ontology>_<distance>
    distances: List[str] = field(default_factory=lambda: ["SimRel"])
    # Random forest (columns/classifier.py)
    rf_max_cells: int = 20000
    rf_trees: int = 300
    rf_folds: int = 5
    n_jobs: int = -1
    seed: int = 42


def dataset_dirs(results_root: Path, datasets: Optional[Sequence[str]] = None) -> dict:
    """{dataset label: results folder} of results/dataset_<x>, e.g. A -> results/dataset_a, A+B -> results/dataset_a+b."""
    found = {p.name[len("dataset_"):].upper(): p for p in sorted(Path(results_root).glob("dataset_*"))
             if p.is_dir()}
    return {d: p for d, p in found.items() if datasets is None or d in datasets}


def genes_hash(genes: Sequence[str]) -> str:
    """Short fingerprint of a gene list, so cached results notice a changed list."""
    return hashlib.sha1(";".join(genes).encode()).hexdigest()[:12]


def load_experiments(results_root: Path, datasets: Optional[Sequence[str]] = None,
                     model_forms: Sequence[str] = DEFAULT_MODEL_FORMS) -> pd.DataFrame:
    """One row per existing gene list, indexed by KEY.

    Besides KEY: method, selection, model and cell_type (gene_sets.py names),
    results_dir (the dataset's results folder), gene_list (path) and genes
    (highest score first). Missing gene lists are left out and counted per
    dataset and method (A+B only has the Geneformer zero-shot ones).
    """
    rows, missing = [], {}
    for dataset, results_dir in dataset_dirs(results_root, datasets).items():
        for method in gs.METHODS:
            meta = gs.METHOD_META[method]
            if meta["model_form"] not in model_forms:
                continue
            for selection in gs.GENE_LIST_PATHS[method]:
                for cell_type in gs.CELL_TYPES:
                    path = gs.gene_list_path(results_dir, method, selection, cell_type)
                    if not path.exists():
                        missing[(dataset, method)] = missing.get((dataset, method), 0) + 1
                        continue
                    df = pd.read_csv(path).sort_values("score", ascending=False, kind="stable")
                    rows.append({
                        "Ranking": RANKING_LABELS[meta["ranking"]],
                        "Score": SCORE_LABELS[meta["score"]],
                        "TopN": TOPN_LABELS[selection],
                        "Model": MODEL_LABELS[meta["model"]],
                        "ModelForm": MODEL_FORM_LABELS[meta["model_form"]],
                        "Dataset": dataset,
                        "CellType": CELL_TYPE_LABELS[cell_type],
                        "method": method, "selection": selection, "model": meta["model"],
                        "cell_type": cell_type, "results_dir": results_dir, "gene_list": path,
                        "genes": df["gene"].astype(str).tolist(),
                    })
    for (dataset, method), n in missing.items():
        print(f"  [macrotable] {dataset} {method}: {n} gene lists missing", flush=True)
    if not rows:
        raise SystemExit(f"No gene lists found under {results_root}/dataset_*")
    return pd.DataFrame(rows).set_index(KEY)
