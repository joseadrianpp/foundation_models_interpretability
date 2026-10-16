"""
Summary statistics over the enrichment CSVs produced by `enrichment.py`:
per (cell_type, method) term counts/IC/intersection size/functional
abundance, and pairwise Mann-Whitney significance of the difference between
methods of the same model.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .gene_sets import CELL_TYPES, MODELS, SELECTIONS, methods_for, model_of, output_dir
from .plots import plot_enrichment_summary

METRICS = ["information_content", "intersection_size", "functional_abundance"]


# Load one method and cell type enrichment CSV and calculate its per-term IC and functional abundance
def _load_terms(enrichment_dir: Path, method: str, cell_type: str) -> pd.DataFrame:
    path = enrichment_dir / method / f"{cell_type}.csv"
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()
    if len(df) == 0:
        return df
    # Frequency-based Information Content: rarer/more specific terms score higher
    df["information_content"] = -np.log2(df["term_size"] / df["effective_domain_size"])
    # Functional abundance (g:Profiler's recall): how much of each term our gene set covers
    df["functional_abundance"] = df["intersection_size"] / df["term_size"]
    return df


# One row per significant term, tagged with method/cell_type, across all pairs
def _build_long_terms(enrichment_dir: Path, methods: list) -> pd.DataFrame:
    # Every combination variable
    frames = []

    # For every method and cell type this for calls load_terms
    for method in methods:
        for cell_type in CELL_TYPES:
            df = _load_terms(enrichment_dir, method, cell_type)
            if len(df) == 0:
                continue
            df = df[METRICS].copy()
            df["method"] = method
            df["cell_type"] = cell_type
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=METRICS + ["method", "cell_type"])
    return pd.concat(frames, ignore_index=True)


# Number of query genes per (method, cell_type), from the enrichment manifest
def _query_sizes(enrichment_dir: Path) -> dict:
    path = enrichment_dir / "manifest.csv"
    try:
        manifest = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return {}
    return {(r.method, r.cell_type): int(r.n_query_genes) for r in manifest.itertuples()}


# One row per (cell_type, method) with the pooled term-count/IC/intersection/functional abundance stats
def _build_summary_table(long_terms: pd.DataFrame, methods: list, sizes: dict) -> pd.DataFrame:
    rows = []
    for method in methods:
        for cell_type in CELL_TYPES:
            sub = long_terms[(long_terms["method"] == method) & (long_terms["cell_type"] == cell_type)]
            rows.append({
                "cell_type": cell_type, "method": method,
                "n_query_genes": sizes.get((method, cell_type), np.nan),
                "n_significant_terms": len(sub),
                "mean_information_content": sub["information_content"].mean() if len(sub) else np.nan,
                "mean_intersection_size": sub["intersection_size"].mean() if len(sub) else np.nan,
                "mean_functional_abundance": sub["functional_abundance"].mean() if len(sub) else np.nan,
            })
    return pd.DataFrame(rows)

# Mann-Whitney U test between every pair of methods of the same model, per
# cell_type and metric. p-values are BH-adjusted within each
# (model, cell_type, metric) family of pairs.
def _build_pairwise_tests(long_terms: pd.DataFrame, methods: list) -> pd.DataFrame:
    rows = []
    for model, cell_type, metric in itertools.product(MODELS, CELL_TYPES, METRICS):
        model_methods = [m for m in methods if model_of(m) == model]
        if len(model_methods) < 2:
            continue
        family = []
        for m1, m2 in itertools.combinations(model_methods, 2):
            v1 = long_terms[(long_terms["cell_type"] == cell_type) & (long_terms["method"] == m1)][metric].dropna()
            v2 = long_terms[(long_terms["cell_type"] == cell_type) & (long_terms["method"] == m2)][metric].dropna()
            # Mann-Whitney needs at least 2 samples per side to be meaningful
            p = np.nan if len(v1) < 2 or len(v2) < 2 else mannwhitneyu(v1, v2, alternative="two-sided").pvalue
            family.append({"model": model, "cell_type": cell_type, "metric": metric,
                           "method_a": m1, "method_b": m2,
                           "n_a": len(v1), "n_b": len(v2), "p_value": p})
        p_values = np.array([r["p_value"] for r in family], dtype=float)
        p_adj = np.full_like(p_values, np.nan)
        tested = np.isfinite(p_values)
        if tested.any():
            p_adj[tested] = multipletests(p_values[tested], method="fdr_bh")[1]
        for r, q in zip(family, p_adj):
            r["p_adj_bh"] = q
        rows += family
    return pd.DataFrame(rows, columns=["model", "cell_type", "metric", "method_a", "method_b",
                                       "n_a", "n_b", "p_value", "p_adj_bh"])


def _summary_for_selection(results_dir: Path, selection: str) -> None:
    methods = methods_for(selection)
    enrichment_dir = output_dir(results_dir, "enrichment", selection)
    out_dir = output_dir(results_dir, "enrichment_summary", selection)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pooled term-count/IC/intersection/functional abundance stats per cell_type and method
    long_terms = _build_long_terms(enrichment_dir, methods)
    summary = _build_summary_table(long_terms, methods, _query_sizes(enrichment_dir))
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"Saved {out_dir / 'summary_table.csv'} ({len(summary)} rows)")

    # Mann-Whitney significance of the difference between methods
    tests = _build_pairwise_tests(long_terms, methods)
    tests.to_csv(out_dir / "pairwise_tests.csv", index=False)
    print(f"Saved {out_dir / 'pairwise_tests.csv'} ({len(tests)} rows)")

    plot_enrichment_summary(results_dir, selection)


def task_enrichment_summary(results_dir: Path, selections=SELECTIONS) -> None:
    for selection in selections:
        _summary_for_selection(results_dir, selection)
