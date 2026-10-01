"""
g:Profiler enrichment of each method's gene set (top-30 or elbow) per cell
type, against g:Profiler's default annotated-genome background.
"""

from pathlib import Path

import pandas as pd
from gprofiler import GProfiler

from .gene_sets import CELL_TYPES, SELECTIONS, load_genes, methods_for, output_dir
from .plots import plot_gene_set_overview

# This are ontologies of gprofiler
GPROFILER_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]

# Function for running the enrichment with default background of g:Profiler
def _run_enrichment(gp: GProfiler, genes: list) -> pd.DataFrame:
    df = gp.profile(
        organism="hsapiens", query=genes, sources=GPROFILER_SOURCES,
        no_iea=True, significance_threshold_method="g_SCS",
        domain_scope="annotated",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # Keep only the terms g:Profiler flagged as significant
    return df[df["significant"] == True].copy()

# Run enrichment for every method and cell type pair of one selection and record a manifest
def _enrichment_for_selection(gp: GProfiler, results_dir: Path, selection: str) -> None:
    out_dir = output_dir(results_dir, "enrichment", selection)
    manifest_rows = []

    # Iterate over methods and cell types
    for method in methods_for(selection):
        for cell_type in CELL_TYPES:
            genes = load_genes(results_dir, method, selection, cell_type)
            if not genes:
                continue
            sig_df = _run_enrichment(gp, genes)

            # Write the significant terms of this (method, cell type) pair
            method_dir = out_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            sig_df.to_csv(method_dir / f"{cell_type}.csv", index=False)

            # Track number of terms counts across all pairs
            # This will be saved in manifest.csv
            manifest_rows.append({
                "method": method, "cell_type": cell_type,
                "n_query_genes": len(genes), "n_significant_terms": len(sig_df),
            })
            print(f"[{selection}/{method}/{cell_type}] {len(genes)} genes -> {len(sig_df)} significant terms")

    # One-row-per-pair summary of how many genes went in and terms came out
    manifest = pd.DataFrame(manifest_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved {out_dir / 'manifest.csv'} ({len(manifest)} rows)")


def task_enrichment(results_dir: Path, selections=SELECTIONS) -> None:
    # Initialization of the GProfiler
    gp = GProfiler(return_dataframe=True)
    for selection in selections:
        _enrichment_for_selection(gp, results_dir, selection)
    # Gene-set sizes and overlap between methods, independent of enrichment
    plot_gene_set_overview(results_dir)
