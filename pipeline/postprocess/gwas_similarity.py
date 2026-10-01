"""
Gene-level GO3 semantic similarity (go3.compare_gene_set_pairs_batch, BMA)
between each method's gene set (top-30 or elbow) and three GWAS Catalog
diseases reference gene sets, per cell type / ontology / distance measure.
"""

from pathlib import Path

import pandas as pd

import go3

from . import _go3_common as g3c
from .gene_sets import CELL_TYPES, SELECTIONS, load_genes, methods_for, output_dir
from .plots import plot_gwas_set_similarity

ONTOLOGIES = ["BP", "CC", "MF"]
GO_SOURCES = {"BP": "GO:BP", "CC": "GO:CC", "MF": "GO:MF"}
DEFAULT_DISTANCES = ["lin", "SimRel", "wang"]


# One entry per (method, cell type) with its gene set, skipping missing pairs
def _collect_entries(results_dir: Path, selection: str) -> list:
    entries = []
    for method in methods_for(selection):
        for cell_type in CELL_TYPES:
            genes = load_genes(results_dir, method, selection, cell_type)
            if genes:
                entries.append({"method": method, "cell_type": cell_type, "genes": genes})
    return entries


# Per reference, genes carrying >=1 GAF annotation in each sub-ontology
def _annotated_counts(refs: dict, out_csv: Path) -> None:
    sym_by_go = g3c.gaf_symbols_by_go_source()
    rows = []
    for key, genes in refs.items():
        gset = set(genes)
        for go_source, syms in sym_by_go.items():
            rows.append({"reference": key, "go_source": go_source, "n_annotated": len(gset & syms)})
        # Also record the reference's total size regardless of GO annotation
        rows.append({"reference": key, "go_source": "total", "n_annotated": len(gset)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# One similarity row per (reference, method, cell_type, ontology, distance),that's the four fors
def _compute(entries: list, refs: dict, counter, distances: list) -> pd.DataFrame:
    rows = []
    for ref_key, ref_genes in refs.items():
        for ont in ONTOLOGIES:
            # Every method/cell_type gene set is compared against the same reference
            pairs = [(e["genes"], ref_genes) for e in entries]
            for dist in distances:
                # One batched go3 call per (ontology, distance) covers every entry
                sims = go3.compare_gene_set_pairs_batch(pairs, ont, dist, "bma", counter)
                for e, s in zip(entries, sims):
                    rows.append({"reference": ref_key, "method": e["method"],
                                 "cell_type": e["cell_type"], "n_genes": len(e["genes"]),
                                 "go_source": GO_SOURCES[ont],
                                 "distance": dist, "similarity": float(s)})
            print(f"[{ref_key}] {GO_SOURCES[ont]} done")
    return pd.DataFrame(rows)

def task_gwas_similarity(results_dir: Path, selections=SELECTIONS) -> None:
    refs_dir = results_dir / "gwas_similarity" / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Fetch (or reuse the cached) GWAS Catalog gene sets for PD/AD/CAD
    refs = g3c.fetch_all_gwas(refs_dir)
    print("Reference sizes:", {k: len(v) for k, v in refs.items()})

    # Loading the GO ontology
    counter = g3c.init_go3()
    # Record how many genes in each reference (i.e., PD/AD/CAD) are annotated in each sub-ontology
    _annotated_counts(refs, refs_dir / "annotated_counts.csv")

    for selection in selections:
        # Load the gene set of every (method, cell type)
        entries = _collect_entries(results_dir, selection)
        print(f"[{selection}] collected {len(entries)} gene sets; distances={DEFAULT_DISTANCES}")

        out_dir = output_dir(results_dir, "gwas_similarity", selection)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = _compute(entries, refs, counter, DEFAULT_DISTANCES)
        df.to_csv(out_dir / "gene_level_similarity_all.csv", index=False)
        print("Saved:", out_dir / "gene_level_similarity_all.csv")

        plot_gwas_set_similarity(results_dir, selection)
