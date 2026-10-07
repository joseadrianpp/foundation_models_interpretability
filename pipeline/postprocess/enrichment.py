"""
g:Profiler enrichment of the top-30 genes per method and cell type, against
g:Profiler's default annotated-genome background.
"""

from pathlib import Path

import pandas as pd
from gprofiler import GProfiler

METHODS = ["attention_zero_shot_scgpt", "attention_lora_scgpt", "ig_lora_scgpt",
           "attention_zero_shot_geneformer", "attention_lora_geneformer", "ig_lora_geneformer",
           "attention_zero_shot_sccello", "attention_lora_sccello", "ig_lora_sccello"]
CELL_TYPES = ["astrocytes", "da_neurons", "microglia", "oligodendrocytes"]
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

# Run enrichment for every method and cell type pair and record a manifest
def task_enrichment(results_dir: Path) -> None:

    # Initialization of the GProfiler
    gp = GProfiler(return_dataframe=True)
    out_dir = results_dir / "enrichment"
    manifest_rows = []

    # Iterate over methods and cell types
    for method in METHODS:
        for cell_type in CELL_TYPES:
            # top30_genes.csv is written by run_tasks.py's attention/ig_lora tasks
            genes_path = results_dir / method / cell_type / "top30_genes.csv"
            if not genes_path.exists():
                print(f"  MISSING {genes_path}")
                continue
            genes = pd.read_csv(genes_path)["gene"].tolist()
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
            print(f"[{method}/{cell_type}] {len(genes)} genes -> {len(sig_df)} significant terms")

    # One-row-per-pair summary of how many genes went in and terms came out
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved manifest.csv ({len(manifest)} rows)")
