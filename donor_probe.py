#!/usr/bin/env python
"""Does a scGPT form separate PD from control in donors it has not seen?

The models' train/val/test split is by cell: every donor is in training and
in test, with one label per donor, so a classifier can score well by
recognising donors. This script trains a linear probe (standardised logistic
regression on the embedding's first 50 principal components) on the cell
embeddings each form's IG task saved
(<results-dir>/dataset_<x>/ig_<form>_scgpt/<cell type>/cell_embeddings.npy,
the <cls> entering the last layer) in two ways:

  cells    the models' own split: trained on the train cells, scored on the test cells
  donors   5-fold cross-validation with whole donors held out (stratified by label)

and reports balanced accuracy and AUC per cell, plus the AUC per donor (the
mean predicted probability of its cells). A large drop from `cells` to
`donors` means the separation comes from recognising donors, not the disease.
It also reports how well the embedding identifies the donor itself
(multinomial probe, cell split, balanced accuracy; chance = 1 / donors).

Forms: zero_shot (the zero-shot + MLP form uses the same embedding) and
step2_5, trained without labels, so both probes are clean. The LoRA encoder
was trained with the labels of every donor, so its `donors` probe is not a
held-out test (shown for reference only); that needs retraining LoRA.

Cells are subsampled to --max-cells per cell type (seed 42).

  python donor_probe.py --results-dir results/scgpt --data-dir data --datasets A B
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SPLITS = ("train", "val", "test")
H5AD = "mixed_split_scgpt/Mixed_allgenes_pc_noXY_noMT_mixed_{split}.h5ad"
FORMS = ("zero_shot", "step2_5", "lora")


def _obs(data_dir: Path) -> pd.DataFrame:
    """Cell type, donor and split of every cell, in the pipeline's order (train, val, test)."""
    parts = []
    for split in SPLITS:
        a = ad.read_h5ad(data_dir / H5AD.format(split=split), backed="r")
        parts.append(a.obs[["cell_type", "donor"]].astype(str).assign(split=split))
        a.file.close()
    obs = pd.concat(parts, ignore_index=True)
    obs["slug"] = obs["cell_type"].map(lambda c: "_".join(c.strip().lower().split()))
    return obs


def _probe():
    return make_pipeline(StandardScaler(), PCA(50, random_state=SEED), LogisticRegression(max_iter=1000))


def _scores(y, p, donors) -> dict:
    by_donor = pd.DataFrame({"y": y, "p": p, "d": donors}).groupby("d").mean()
    return {"bal_acc": balanced_accuracy_score(y, p > 0.5), "auc": roc_auc_score(y, p),
            "donor_auc": roc_auc_score(by_donor["y"], by_donor["p"]) if by_donor["y"].nunique() == 2 else np.nan}


def run(results_dir: Path, data_dir: Path, dataset: str, max_cells: int) -> pd.DataFrame:
    obs = _obs(data_dir / f"dataset_{dataset.lower()}")
    rows = []
    for slug, cells in obs.groupby("slug", sort=True):
        ig_dir = results_dir / f"dataset_{dataset.lower()}"
        with np.load(ig_dir / "ig_zero_shot_scgpt" / slug / "cell_scores.npz") as z:
            labels = z["labels"]
        if labels.size != len(cells):
            raise SystemExit(f"{slug}: {labels.size} labelled cells, {len(cells)} in the h5ad files")
        keep = np.arange(len(cells))
        if keep.size > max_cells:
            keep = np.sort(np.random.default_rng(SEED).choice(keep, max_cells, replace=False))
        y, donors, split = labels[keep], cells["donor"].to_numpy()[keep], cells["split"].to_numpy()[keep]
        for form in FORMS:
            x = np.load(ig_dir / f"ig_{form}_scgpt" / slug / "cell_embeddings.npy")[keep]
            # The models' split: train cells -> test cells
            train, test = split == "train", split == "test"
            p = _probe().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
            rows.append({"dataset": dataset, "cell_type": slug, "form": form, "split": "cells",
                         "n_donors": len(set(donors)), **_scores(y[test], p, donors[test])})
            # Whole donors held out
            p = np.zeros(y.size)
            for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y, donors):
                p[te] = _probe().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
            rows.append({"dataset": dataset, "cell_type": slug, "form": form, "split": "donors",
                         "n_donors": len(set(donors)), **_scores(y, p, donors)})
            # How well the embedding identifies the donor (cell split)
            d = _probe().fit(x[train], donors[train]).predict(x[test])
            rows.append({"dataset": dataset, "cell_type": slug, "form": form, "split": "donor_id",
                         "n_donors": len(set(donors)), "bal_acc": balanced_accuracy_score(donors[test], d)})
            print(f"  {dataset} {slug} {form}: done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results/scgpt"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--datasets", nargs="+", default=["A", "B"])
    p.add_argument("--max-cells", type=int, default=20000)
    args = p.parse_args()
    out = pd.concat([run(args.results_dir, args.data_dir, d, args.max_cells) for d in args.datasets])
    out_dir = args.results_dir / "donor_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "donor_probe.csv", index=False)
    summary = out.groupby(["dataset", "form", "split"])[["bal_acc", "auc", "donor_auc"]].mean()
    summary.to_csv(out_dir / "summary.csv")
    print(f"\n# Linear probe on the cell embeddings (mean over cell types)\n{summary.round(3).to_string()}", flush=True)


if __name__ == "__main__":
    main()
