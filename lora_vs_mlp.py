#!/usr/bin/env python
"""Does LoRA add PD information that zero-shot+MLP lacks? Exclusive genes and per-cell FGCell.

Two paired comparisons of the scGPT classifier forms (plan docs/plan_simpd_lora_2026-11.md, 6.3):

1. Per-cell FGCell. evaluate_attributions.py scores the same cells (attribution_eval/cells.csv)
   with both forms; FGCell of a cell = 1 - deletion AUC of its own signed IG order (towards the true
   class, highest first) / AUC of a random order (attribution_eval/scgpt_<form>/cells/<cell type>.csv, method "ig"). Per dataset x cell
   type, and pooled: two-sided Wilcoxon over the cells (LoRA - MLP) and over the donors (mean
   per donor, since the cells of a donor are not independent).

2. Exclusive genes. For each dataset x cell type x cut-off (top30 / z-score / elbow) the pooled
   |IG| gene sets of the two forms split into LoRA-only, MLP-only and shared genes. Each part
   with at least --min-genes genes gets SimPD and p_expr under --measures (default M0, the
   current SimPD, and M10_resnik_max, the measure most favourable to LoRA in
   simpd_measures/mlp_vs_lora.csv) against the expr null of simpd_measures.py (same number of
   genes per detection decile of the cell type's universe, same seeds). The direct test:
   SimPD(LoRA-only) - SimPD(MLP-only) against --perms random splits of their union into two
   groups of the same sizes (two-sided p). Also the number of each part's genes that are in a
   PD reference or in a control-disease reference.

Outputs in <results-root>/lora_vs_mlp/:
  fgcell_cells.csv      dataset, cell_type (or "all"), n_cells, mean FGCell per form,
                        median difference, fraction of cells where LoRA is higher, Wilcoxon p
                        over cells and over donors
  exclusive_genes.csv   one row per measure x dataset x cell type x cut-off: set sizes,
                        overlap (shared genes / distinct genes), PD / control reference hits,
                        SimPD and p_expr per part,
                        the LoRA-only minus MLP-only difference and its permutation p
  exclusive_summary.csv per measure: fraction of parts with p_expr < 0.05, median p_expr,
                        Wilcoxon of log10 p (LoRA-only vs MLP-only) over the pairs, and how
                        many permutation tests favour each form at p < 0.05

  python lora_vs_mlp.py --results-root results/scgpt --datasets A B
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pipeline.postprocess import _go3_common as g3c
from pipeline.postprocess import gene_sets as gs
from simpd_baseline import N_BINS, SELECTION_DIRS, SEED, SIGN, _detection_bins, _universe
from simpd_measures import (MEASURES, PD_REFS, _bh, _make_scorer, _expr_picks, _gene_sets, _load_mats, _p, _refs,
                            _simpd)

FORMS = ("mlp", "lora")
PARTS = ("lora_only", "mlp_only", "shared")
CONTROL_REFS = [r for r in SIGN if r not in PD_REFS]


def _wilcoxon(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    return float(wilcoxon(d, zero_method="pratt").pvalue) if len(d) and d.any() else 1.0


# ---------------------------------------------------------------------------
# 1. Per-cell FGCell
# ---------------------------------------------------------------------------

def _fgcell_cells(results_root: Path, datasets: list) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        base = results_root / f"dataset_{ds.lower()}" / "attribution_eval"
        for ct in gs.CELL_TYPES:
            per_form = []
            for form in FORMS:
                path = base / f"scgpt_{form}" / "cells" / f"{ct}.csv"
                if not path.exists():
                    break
                c = pd.read_csv(path)
                c = c[c["method"] == "ig"][["cell", "donor", "fg"]].rename(columns={"fg": form})
                per_form.append(c.set_index(["cell", "donor"]))
            if len(per_form) == len(FORMS):
                m = per_form[0].join(per_form[1], how="inner").reset_index()
                m["dataset"], m["cell_type"] = ds, ct
                frames.append(m)
    cells = pd.concat(frames, ignore_index=True)
    rows = []
    groups = [(ds, ct, g) for (ds, ct), g in cells.groupby(["dataset", "cell_type"], sort=False)]
    groups += [(ds, "all", g) for ds, g in cells.groupby("dataset", sort=False)]
    groups += [("all", "all", cells)]
    for ds, ct, g in groups:
        d = g["lora"] - g["mlp"]
        # Donor level: the same donor's cells share most of their signal
        donors = g.groupby(["dataset", "donor"])[["mlp", "lora"]].mean()
        rows.append({"dataset": ds, "cell_type": ct, "n_cells": len(g), "mean_mlp": g["mlp"].mean(),
                     "mean_lora": g["lora"].mean(), "median_diff": d.median(),
                     "frac_lora_higher": float((d > 0).mean()), "wilcoxon_p_cells": _wilcoxon(d),
                     "n_donors": len(donors), "frac_donors_lora_higher": float((donors["lora"] > donors["mlp"]).mean()),
                     "wilcoxon_p_donors": _wilcoxon(donors["lora"] - donors["mlp"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Exclusive genes
# ---------------------------------------------------------------------------

def _parts(mlp: list, lora: list) -> dict:
    m, l = set(mlp), set(lora)
    return {"lora_only": sorted(l - m), "mlp_only": sorted(m - l), "shared": sorted(m & l)}


def _exclusive(args) -> pd.DataFrame:
    data = {}
    for ds in args.datasets:
        results_dir = args.results_root / f"dataset_{ds.lower()}"
        refs = _refs(results_dir)
        sets = {(s[1], s[2]): {} for s in _gene_sets(results_dir, list(FORMS))}
        for form, selection, ct, genes in _gene_sets(results_dir, list(FORMS)):
            sets[(selection, ct)][form] = genes
        data[ds] = {"dir": results_dir, "refs": refs, "sets": sets,
                    "universes": {ct: _universe(results_dir, ct) for ct in gs.CELL_TYPES}}
    # The same gene and reference lists as simpd_measures.py, so the Resnik scale is identical
    all_genes = sorted(set().union(*[set().union(*d["universes"].values()) for d in data.values()]))
    all_refs = sorted({g for d in data.values() for genes in d["refs"].values() for g in genes})
    counter = g3c.init_go3(args.threads)

    rows, check = [], []
    for m in args.measures:
        dist, gene, how, onts, normalise = MEASURES[m]
        mats = _load_mats(dist, gene, onts, all_genes, all_refs, counter, args.results_root / "go3_pair_cache")
        scale = {o: float(np.nanmax(mats[o].values)) if normalise else 1.0 for o in onts}
        print(f"\n# {m}", flush=True)
        for ds, d in data.items():
            pd_genes = set().union(*[set(d["refs"][r]) for r in PD_REFS])
            ctrl_genes = set().union(*[set(d["refs"][r]) for r in CONTROL_REFS])
            for i, ct in enumerate(gs.CELL_TYPES):
                universe = d["universes"][ct]
                index = {g: k for k, g in enumerate(universe)}
                scorer = _make_scorer(universe, d["refs"], mats, scale)
                bins = _detection_bins(d["dir"], ct, universe)
                members = [np.where(bins == b)[0] for b in range(N_BINS)]

                def score(genes):
                    pick = np.array([index[g] for g in genes if g in index])
                    return pick, float(_simpd(scorer.per_ref(pick[None, :], how, onts))[0])

                for s_i, selection in enumerate(SELECTION_DIRS):
                    forms = d["sets"].get((selection, ct), {})
                    if set(forms) != set(FORMS):
                        continue
                    # Regression check against simpd_measures/observed.csv
                    for form in FORMS:
                        check.append({"measure": m, "dataset": ds, "form": form, "cell_type": ct,
                                      "selection": selection, "SimPD_here": score(forms[form])[1]})
                    parts = _parts(forms["mlp"], forms["lora"])
                    row = {"measure": m, "dataset": ds, "cell_type": ct, "selection": selection,
                           "n_mlp": len(forms["mlp"]), "n_lora": len(forms["lora"]),
                           "overlap": len(parts["shared"]) / len(set(forms["mlp"]) | set(forms["lora"]))}
                    picks = {}
                    for part, genes in parts.items():
                        row[f"n_{part}"] = len(genes)
                        row[f"pd_hits_{part}"] = len(set(genes) & pd_genes)
                        row[f"ctrl_hits_{part}"] = len(set(genes) & ctrl_genes)
                        row[f"genes_{part}"] = " ".join(genes)
                        if len(genes) < args.min_genes:
                            row[f"SimPD_{part}"], row[f"p_expr_{part}"] = np.nan, np.nan
                            continue
                        pick, obs = score(genes)
                        picks[part] = pick
                        counts = np.bincount(bins[pick], minlength=N_BINS)
                        null = _simpd(scorer.per_ref(_expr_picks(ds, i, members, counts, args.draws), how, onts))
                        row[f"SimPD_{part}"], row[f"p_expr_{part}"] = obs, _p(obs, null)
                    # LoRA-only vs MLP-only: random splits of their union, same sizes
                    if "lora_only" in picks and "mlp_only" in picks:
                        pool = np.concatenate([picks["lora_only"], picks["mlp_only"]])
                        n_l = picks["lora_only"].size
                        obs = row["SimPD_lora_only"] - row["SimPD_mlp_only"]
                        rng = np.random.default_rng([SEED, ord(ds.upper()), i, 10, s_i])
                        perm = np.stack([rng.permutation(pool) for _ in range(args.perms)])
                        null = (_simpd(scorer.per_ref(perm[:, :n_l], how, onts))
                                - _simpd(scorer.per_ref(perm[:, n_l:], how, onts)))
                        row["diff_lora_minus_mlp"] = obs
                        row["perm_p_lora_higher"] = _p(obs, null)
                        row["perm_p_mlp_higher"] = _p(-obs, -null)
                        row["perm_p_two_sided"] = min(1.0, 2 * min(row["perm_p_lora_higher"], row["perm_p_mlp_higher"]))
                    rows.append(row)
                print(f"  {ds} {ct} done", flush=True)
        del mats

    out = pd.DataFrame(rows)
    for part in PARTS:
        out[f"q_expr_{part}"] = out.groupby("measure")[f"p_expr_{part}"].transform(_bh)
    # The full sets must reproduce simpd_measures.py
    obs_path = args.results_root / "simpd_measures" / "observed.csv"
    if obs_path.exists():
        ref = pd.read_csv(obs_path)
        c = pd.DataFrame(check).merge(ref, on=["measure", "dataset", "form", "cell_type", "selection"])
        print(f"\nregression vs simpd_measures/observed.csv: {len(c)} sets, "
              f"max |diff| = {np.abs(c['SimPD_here'] - c['SimPD']).max():.2e}", flush=True)
    return out


def _exclusive_summary(ex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in ex.groupby("measure", sort=False):
        row = {"measure": m, "n_pairs": len(g), "median_overlap": g["overlap"].median()}
        for part in PARTS:
            p = g[f"p_expr_{part}"]
            row[f"{part}_scored"] = int(p.notna().sum())
            row[f"{part}_frac_p05"] = float((p < 0.05).sum() / max(1, p.notna().sum()))
            row[f"{part}_median_p"] = float(p.median())
            row[f"{part}_pd_hits"] = int(g[f"pd_hits_{part}"].sum())
            row[f"{part}_ctrl_hits"] = int(g[f"ctrl_hits_{part}"].sum())
        # Positive: LoRA's exclusive genes more significant than MLP's
        both = g.dropna(subset=["p_expr_lora_only", "p_expr_mlp_only"])
        d = np.log10(both["p_expr_mlp_only"]) - np.log10(both["p_expr_lora_only"])
        row["n_both_scored"] = len(both)
        row["lora_only_better"] = int((d > 0).sum())
        row["mlp_only_better"] = int((d < 0).sum())
        row["wilcoxon_p_log10p"] = _wilcoxon(d)
        row["perm_lora_higher_p05"] = int((g["perm_p_lora_higher"] < 0.05).sum())
        row["perm_mlp_higher_p05"] = int((g["perm_p_mlp_higher"] < 0.05).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("results/scgpt"))
    p.add_argument("--datasets", nargs="+", default=["A", "B"])
    p.add_argument("--measures", nargs="+", default=["M0_simrel_bma", "M10_resnik_max"], choices=list(MEASURES))
    p.add_argument("--draws", type=int, default=1000, help="expr null draws per gene-set part.")
    p.add_argument("--perms", type=int, default=1000, help="Random splits for LoRA-only vs MLP-only.")
    p.add_argument("--min-genes", type=int, default=3, help="Smaller parts are not scored.")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <results-root>/lora_vs_mlp.")
    args = p.parse_args()
    out_dir = args.out_dir or args.results_root / "lora_vs_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    fg = _fgcell_cells(args.results_root, args.datasets)
    fg.to_csv(out_dir / "fgcell_cells.csv", index=False)
    print(f"# Per-cell FGCell, LoRA vs zero-shot+MLP\n{fg.round(4).to_string(index=False)}", flush=True)

    ex = _exclusive(args)
    ex.to_csv(out_dir / "exclusive_genes.csv", index=False)
    summary = _exclusive_summary(ex)
    summary.to_csv(out_dir / "exclusive_summary.csv", index=False)
    cols = ["measure", "dataset", "cell_type", "selection", "n_mlp", "n_lora", "overlap", "n_lora_only",
            "n_mlp_only", "pd_hits_lora_only", "pd_hits_mlp_only", "p_expr_lora_only", "p_expr_mlp_only",
            "p_expr_shared", "perm_p_two_sided"]
    print(f"\n# Exclusive genes\n{ex[[c for c in cols if c in ex]].round(3).to_string(index=False)}", flush=True)
    print(f"\n# Summary\n{summary.round(3).T.to_string()}", flush=True)
    json.dump({"datasets": args.datasets, "measures": args.measures, "draws": args.draws, "perms": args.perms,
               "min_genes": args.min_genes, "seed": SEED}, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"\noutputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
