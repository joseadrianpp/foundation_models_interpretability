#!/usr/bin/env python
"""Localisation of the per-cell explanations on Parkinson's genes (Quantus' localisation family).

Quantus' localisation metrics score an explanation against a ground-truth
mask of the input. There is no per-cell ground truth here, so the mask is a
reference gene set: in every cell, the genes it expresses that are in the PD
references (GWAS Catalog PD, PanelApp, ParkinsonsUK-UCL, as SimPD), and as a
contrast those in each control disease (Alzheimer, chronic kidney disease,
coronary artery disease; GWAS Catalog). Per cell and explanation:

  auc      ROC AUC of the attribution separating the mask's genes from the
           cell's other genes (quantus.AUC); 0.5 = no preference
  topk     share of the 30 genes with the largest attribution that are in
           the mask, over the share of the cell's genes in it
           (quantus.TopKIntersection, scaled by the mask's prevalence); 1 = chance

and the PD - control difference of each (mean over the three control
diseases), the per-cell analogue of SimPD. The cells are those of the FG
evaluation (attribution_eval/cells.csv: 200 per cell type, correctly
classified by both classifiers, spread over donors); the explanations are the
pipeline's per-cell ones: ig (signed, towards the true class for a
classifier; SIGnature's for an encoder), |ig|, rollout and gradrollout (when
they ran) and
random (seed 42 + cell).

Outputs in <results-dir>/dataset_<x>/localisation/: cells.csv (one row per
cell, form, explanation and reference) and summary.csv.

  python evaluate_localisation.py --dataset A --results-dir results/scgpt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_tasks as rt

SEED = 42
TOP_K = 30
PD_REFS = ("parkinson", "panelapp", "parkinsonsuk_ucl")
CONTROL_REFS = ("alzheimer", "kidney", "cad")


def _per_cell(path: Path):
    """{cell index: (token ids, values)} of a pipeline cell_scores.npz (IG or rollout)."""
    with np.load(path) as z:
        cells, gids, vals = z["token_cells"], z["token_gids"], z["token_ig"]
    ptr = np.searchsorted(cells, np.arange(cells.max() + 2))
    return lambda i: (gids[ptr[i]:ptr[i + 1]], vals[ptr[i]:ptr[i + 1]])


def _scores(a: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0 or mask.all():
        return {"auc": np.nan, "topk": np.nan, "n_mask": int(mask.sum())}
    top = np.argsort(-a, kind="stable")[:TOP_K]
    return {"auc": roc_auc_score(mask, a), "topk": mask[top].mean() / mask.mean(), "n_mask": int(mask.sum())}


def run(dataset: str, results_root: Path) -> None:
    results_dir = results_root / f"dataset_{dataset.lower()}"
    names = rt._load_scgpt_gene_names(rt.DATASET_DIRS[dataset])
    refs_dir = results_dir / "gwas_similarity" / "refs"
    refs = {k: set(pd.read_csv(refs_dir / f"{k}_genes.csv")["gene"].astype(str)) for k in PD_REFS + CONTROL_REFS}
    masks = {"PD": set().union(*(refs[k] for k in PD_REFS)), **{k: refs[k] for k in CONTROL_REFS}}
    cells = pd.read_csv(results_dir / "attribution_eval" / "cells.csv")

    rows = []
    for form in rt.FORMS:
        for slug, sel in cells.groupby("cell_type"):
            ig = _per_cell(results_dir / f"ig_{form}_scgpt" / slug / "cell_scores.npz")
            extra = {kind: _per_cell(results_dir / f"{kind}_{form}_scgpt" / slug / "cell_scores.npz")
                     for kind in ("rollout", "gradrollout")
                     if (results_dir / f"{kind}_{form}_scgpt" / slug / "cell_scores.npz").exists()}
            for r in sel.itertuples(index=False):
                gids, a = ig(int(r.cell))
                genes = np.array([names[int(g)] for g in gids])
                if form in rt.CLASSIFIER_FORMS and r.label == 0:
                    a = -a
                expl = {"ig": a, "|ig|": np.abs(a),
                        "random": np.random.default_rng(SEED + int(r.cell)).random(a.size)}
                for kind, values_of in extra.items():
                    k_gids, k_a = values_of(int(r.cell))
                    if np.array_equal(k_gids, gids):
                        expl[kind] = k_a
                for ref, gene_set in masks.items():
                    mask = np.isin(genes, list(gene_set))
                    for method, values in expl.items():
                        rows.append({"form": form, "cell_type": slug, "cell": r.cell, "label": r.label,
                                     "method": method, "reference": ref, **_scores(values, mask)})
        print(f"  {dataset} {form}: done", flush=True)

    out_dir = results_dir / "localisation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cells.csv", index=False)
    # PD and the mean of the control diseases; their difference per cell (the per-cell SimPD)
    df["group"] = np.where(df["reference"] == "PD", "PD", "control")
    per_cell = df.groupby(["form", "cell_type", "cell", "method", "group"])[["auc", "topk"]].mean().unstack("group")
    per_cell.columns = [f"{m}_{g}" for m, g in per_cell.columns]
    per_cell["auc_diff"] = per_cell["auc_PD"] - per_cell["auc_control"]
    per_cell["topk_diff"] = per_cell["topk_PD"] - per_cell["topk_control"]
    summary = per_cell.groupby(["form", "method"]).mean()
    summary.to_csv(out_dir / "summary.csv")
    print(f"\n# {dataset}: localisation on PD genes (mean over cells)\n{summary.round(3).to_string()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--results-dir", type=Path, default=Path("results/scgpt"))
    args = p.parse_args()
    for d in args.dataset:
        run(d, args.results_dir)


if __name__ == "__main__":
    main()
