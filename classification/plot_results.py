#!/usr/bin/env python
"""
Table and figures of the MLP results of every dataset and model, the ones
presenting_results.md explains how to read and report.

  summary.csv            one row per dataset and model: test balanced accuracy
                         with its 95% CI, sensitivity, specificity, ROC AUC and
                         test donors classified correctly. With several seeds
                         (train_mlp.py --seed), the mean over the seeds and the
                         95% CI of that mean; with one, the CI over the cells
  balanced_accuracy      test balanced accuracy and 95% CI per model and dataset
  test_donors            PD probability of the test cells, donor by donor
  training               loss and validation balanced accuracy per epoch
  umap (--umap)          UMAP of the embeddings, by diagnosis and by test donor

Writes in results/figures/ (PNG and PDF). Run in scgpt_env after train_mlp.py:

    python classification/plot_results.py [--umap]
"""

import argparse
import json

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

import config as C

MODEL_LABELS = {"scgpt": "scGPT", "geneformer": "Geneformer", "sccello": "scCello"}
# Colour-blind safe categorical colours, always in this order: one per model,
# and the first two for the two diagnoses
MODEL_COLORS = {"scgpt": "#2a78d6", "geneformer": "#eb6834", "sccello": "#1baf7a"}
DIAGNOSIS_COLORS = {"control": "#2a78d6", C.PD_LABEL: "#eb6834"}
# The four test donors (two PD, two control) in the UMAP
TEST_DONOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLOR_BY_LABEL = {MODEL_LABELS[m]: c for m, c in MODEL_COLORS.items()}
OTHER_COLOR = "#c8c7c1"
OUT_DIR = C.RESULTS_DIR / "figures"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "pdf.fonttype": 42})


# Every (dataset, model) with a trained MLP and the seeds it was trained with
def finished_runs() -> dict:
    runs = {}
    for dataset in C.DATASET_DIRS:
        for model in C.MODELS:
            seeds = sorted(C.mlp_dir(dataset, model).glob("seed_*/test_metrics.json"))
            if seeds:
                runs[dataset, model] = [p.parent for p in seeds]
    return runs


def summary_table(runs: dict) -> pd.DataFrame:
    rows = []
    for (dataset, model), seed_dirs in runs.items():
        tests = [json.loads((d / "test_metrics.json").read_text())["test"] for d in seed_dirs]
        ba = np.array([t["balanced_accuracy"] for t in tests])
        # One seed: the CI over the cells. Several: the CI of the mean over the seeds
        if len(tests) == 1:
            lo, hi = tests[0]["balanced_accuracy_ci95"]
        else:
            lo, hi = stats.t.interval(0.95, len(ba) - 1, loc=ba.mean(), scale=stats.sem(ba))
        rows.append({
            "dataset": dataset, "model": MODEL_LABELS[model], "n_seeds": len(tests),
            "test_cells_pd": tests[0]["n_cells_pd"], "test_cells_control": tests[0]["n_cells_control"],
            "balanced_accuracy": ba.mean(), "ci95_low": lo, "ci95_high": hi,
            "ci_over": "cells" if len(tests) == 1 else "seeds",
            **{k: np.mean([t[k] for t in tests]) for k in ("sensitivity", "specificity", "roc_auc")},
            "donors_correct": ", ".join(t["donors_correct"] for t in tests),
        })
    return pd.DataFrame(rows)


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure {OUT_DIR / name}.png")


# One point per model with its 95% CI, a panel per dataset, and the chance level (0.5)
def plot_balanced_accuracy(summary: pd.DataFrame) -> None:
    datasets = sorted(summary["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3), sharey=True, squeeze=False)
    for ax, dataset, letter in zip(axes[0], datasets, "abcdef"):
        rows = summary[summary["dataset"] == dataset].reset_index(drop=True)
        for x, row in rows.iterrows():
            ax.errorbar(x, row["balanced_accuracy"],
                        yerr=[[row["balanced_accuracy"] - row["ci95_low"]], [row["ci95_high"] - row["balanced_accuracy"]]],
                        fmt="o", ms=7, color=COLOR_BY_LABEL[row["model"]], capsize=4, lw=2)
        ax.axhline(0.5, ls="--", lw=1, color="#52514e")
        ax.set_xticks(range(len(rows)), rows["model"])
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"{letter}) Dataset {dataset}", loc="left")
    axes[0, 0].set_ylabel("Test balanced accuracy (95% CI)")
    save(fig, "balanced_accuracy")


# The PD probability of the cells of each test donor: a classifier that has
# learned the disease puts the PD donors above 0.5 and the controls below
def plot_test_donors(runs: dict) -> None:
    datasets = sorted({d for d, _ in runs})
    fig, axes = plt.subplots(len(datasets), len(C.MODELS), figsize=(3 * len(C.MODELS), 3.4 * len(datasets)),
                             sharey=True, squeeze=False, layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(C.MODELS):
            ax = axes[i, j]
            if (dataset, model) not in runs:
                ax.set_axis_off()
                continue
            pred = pd.read_csv(runs[dataset, model][0] / "predictions.csv")
            test = pred[pred[C.SPLIT_COL] == "test"].sort_values([C.LABEL_COL, C.DONOR_COL])
            donors = test.drop_duplicates(C.DONOR_COL)[[C.DONOR_COL, C.LABEL_COL]].to_numpy()
            boxes = ax.boxplot([test.loc[test[C.DONOR_COL] == d, "prob_pd"] for d, _ in donors],
                               patch_artist=True, showfliers=False, widths=0.6, medianprops={"color": "#0b0b0b"})
            for box, (_, diagnosis) in zip(boxes["boxes"], donors):
                box.set(facecolor=DIAGNOSIS_COLORS[diagnosis], edgecolor="#0b0b0b", linewidth=0.8)
            counts = test[C.DONOR_COL].value_counts()
            ax.set_xticks(range(1, len(donors) + 1),
                          [f"{d}\n{diagnosis}\nn={counts[d]}" for d, diagnosis in donors], fontsize=7)
            ax.axhline(0.5, ls="--", lw=1, color="#52514e")
            ax.set_ylim(0, 1)
            ax.set_title(f"{MODEL_LABELS[model]}, Dataset {dataset}", loc="left")
        axes[i, 0].set_ylabel("PD probability of each cell")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DIAGNOSIS_COLORS.values()]
    fig.legend(handles, list(DIAGNOSIS_COLORS), loc="outside upper center", ncol=2, frameon=False)
    save(fig, "test_donors")


# Loss and validation balanced accuracy in separate rows (two quantities, two
# axes), with the kept epoch (lowest validation loss) marked
def plot_training(runs: dict) -> None:
    keys = list(runs)
    fig, axes = plt.subplots(2, len(keys), figsize=(2.6 * len(keys), 4.4), sharex="col", squeeze=False)
    for j, (dataset, model) in enumerate(keys):
        history = pd.read_csv(runs[dataset, model][0] / "training_history.csv")
        kept = history.loc[history["kept"], "epoch"].iloc[-1]
        axes[0, j].plot(history["epoch"], history["train_loss"], lw=2, ls="--", color="#52514e")
        axes[0, j].plot(history["epoch"], history["val_loss"], lw=2, color=MODEL_COLORS[model])
        axes[1, j].plot(history["epoch"], history["val_balanced_accuracy"], lw=2, marker="o", ms=4,
                        color=MODEL_COLORS[model])
        for ax in axes[:, j]:
            ax.axvline(kept, ls=":", lw=1, color="#0b0b0b")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0, j].set_title(f"{MODEL_LABELS[model]}, {dataset}", loc="left")
        axes[1, j].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Validation\nbalanced accuracy")
    axes[0, 0].legend([plt.Line2D([], [], lw=2, ls="--", color="#52514e"), plt.Line2D([], [], lw=2, color="#0b0b0b")],
                      ["train", "validation"], frameon=False)
    save(fig, "training")


# UMAP of each model's embedding: coloured by diagnosis, and by test donor
# (the rest in grey). If the cells group by donor more than by diagnosis, the
# embedding mostly tells individuals apart, which a donor split exposes
def plot_umap(runs: dict) -> None:
    keys = list(runs)
    fig, axes = plt.subplots(len(keys), 2, figsize=(7, 3.2 * len(keys)), squeeze=False)
    for i, (dataset, model) in enumerate(keys):
        emb = ad.read_h5ad(C.embeddings_path(dataset, model))
        sc.pp.neighbors(emb, use_rep="X", random_state=C.SEED)
        sc.tl.umap(emb, random_state=C.SEED)
        xy, obs = emb.obsm["X_umap"], emb.obs
        order = np.random.default_rng(C.SEED).permutation(len(obs))
        colors = obs[C.LABEL_COL].astype(str).map(DIAGNOSIS_COLORS).to_numpy()
        axes[i, 0].scatter(*xy[order].T, c=colors[order], s=1, linewidths=0, rasterized=True)
        test_donors = sorted(obs.loc[obs[C.SPLIT_COL] == "test", C.DONOR_COL].astype(str).unique())
        palette = dict(zip(test_donors, TEST_DONOR_COLORS))
        diagnosis = obs.drop_duplicates(C.DONOR_COL).set_index(C.DONOR_COL)[C.LABEL_COL].astype(str)
        donor_colors = obs[C.DONOR_COL].astype(str).map(palette).fillna(OTHER_COLOR).to_numpy()
        is_test = (obs[C.SPLIT_COL] == "test").to_numpy()
        axes[i, 1].scatter(*xy[~is_test].T, c=OTHER_COLOR, s=1, linewidths=0, rasterized=True)
        axes[i, 1].scatter(*xy[is_test].T, c=donor_colors[is_test], s=1, linewidths=0, rasterized=True)
        donor_legend = {f"{d} ({diagnosis[d]})": c for d, c in palette.items()}
        for ax, title, legend in ((axes[i, 0], "diagnosis", DIAGNOSIS_COLORS), (axes[i, 1], "test donor", donor_legend)):
            ax.set_title(f"{MODEL_LABELS[model]}, {dataset}: {title}", loc="left")
            ax.set_xticks([]), ax.set_yticks([])
            ax.legend([plt.Line2D([], [], ls="", marker="o", color=c) for c in legend.values()],
                      list(legend), frameon=False, fontsize=7, markerscale=1, loc="best")
    save(fig, "umap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--umap", action="store_true", help="also the UMAPs (a few minutes)")
    args = parser.parse_args()

    runs = finished_runs()
    if not runs:
        raise SystemExit("No MLP results yet: run train_mlp.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summary_table(runs)
    summary.to_csv(OUT_DIR / "summary.csv", index=False, float_format="%.3f")
    print(summary.to_string(index=False, float_format="%.3f"))

    plot_balanced_accuracy(summary)
    plot_test_donors(runs)
    plot_training(runs)
    if args.umap:
        plot_umap(runs)


if __name__ == "__main__":
    main()
