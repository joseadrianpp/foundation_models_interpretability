"""
Summary statistics over the enrichment CSVs produced by `enrichment.py`:
per (cell_type, method) term counts/IC/intersection size/functional
abundance, and pairwise Mann-Whitney significance of the difference between
methods.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .enrichment import CELL_TYPES, METHODS
from .plots import plot_enrichment_summary


# Load one method and cell type enrichment CSV and calculate its per-term IC and functional abundance
def _load_terms(results_dir: Path, method: str, cell_type: str) -> pd.DataFrame:
    path = results_dir / "enrichment" / method / f"{cell_type}.csv"
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()
    if len(df) == 0:
        return df
    # Frequency-based Information Content: rarer/more specific terms score higher
    df["information_content"] = -np.log2(df["term_size"] / df["effective_domain_size"])
    # Functional abundance (g:Profiler's recall): how much of each term our top-30 genes cover
    df["functional_abundance"] = df["intersection_size"] / df["term_size"]
    return df


# One row per significant term, tagged with method/cell_type, across all pairs
def _build_long_terms(results_dir: Path) -> pd.DataFrame:
    # Every combination variable
    frames = []

    # For every method and cell type this for calls load_terms
    for method in METHODS:
        for cell_type in CELL_TYPES:
            df = _load_terms(results_dir, method, cell_type)
            if len(df) == 0:
                continue
            df = df[["information_content", "intersection_size", "functional_abundance"]].copy()
            df["method"] = method
            df["cell_type"] = cell_type
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# One row per (cell_type, method) with the pooled term-count/IC/intersection/functional abundance stats
def _build_summary_table(long_terms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for cell_type in CELL_TYPES:
            sub = long_terms[(long_terms["method"] == method) & (long_terms["cell_type"] == cell_type)]
            rows.append({
                "cell_type": cell_type, "method": method,
                "n_significant_terms": len(sub),
                "mean_information_content": sub["information_content"].mean() if len(sub) else np.nan,
                "mean_intersection_size": sub["intersection_size"].mean() if len(sub) else np.nan,
                "mean_functional_abundance": sub["functional_abundance"].mean() if len(sub) else np.nan,
            })
    return pd.DataFrame(rows)

# Mann-Whitney U test between every pair of methods, per cell_type and metric
def _build_pairwise_tests(long_terms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_type in CELL_TYPES:
        for metric in ["information_content", "intersection_size", "functional_abundance"]:
            # Every pair of methods, 6 comparisons per (cell_type, metric) with 4 methods
            for m1, m2 in itertools.combinations(METHODS, 2):
                v1 = long_terms[(long_terms["cell_type"] == cell_type) & (long_terms["method"] == m1)][metric].dropna()
                v2 = long_terms[(long_terms["cell_type"] == cell_type) & (long_terms["method"] == m2)][metric].dropna()
                # Mann-Whitney needs at least 2 samples per side to be meaningful
                p = np.nan if len(v1) < 2 or len(v2) < 2 else mannwhitneyu(v1, v2, alternative="two-sided").pvalue
                rows.append({"cell_type": cell_type, "metric": metric,
                             "method_a": m1, "method_b": m2,
                             "n_a": len(v1), "n_b": len(v2), "p_value": p})
    return pd.DataFrame(rows)


def task_enrichment_summary(results_dir: Path) -> None:
    out_dir = results_dir / "enrichment_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pooled term-count/IC/intersection stats per cell_type and method
    long_terms = _build_long_terms(results_dir)
    summary = _build_summary_table(long_terms)
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"Saved summary_table.csv ({len(summary)} rows)")

    # Mann-Whitney significance of the difference between methods
    tests = _build_pairwise_tests(long_terms)
    tests.to_csv(out_dir / "pairwise_tests.csv", index=False)
    print(f"Saved pairwise_tests.csv ({len(tests)} rows)")

    plot_enrichment_summary(results_dir)
