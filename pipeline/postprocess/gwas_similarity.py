"""
Gene-level GO3 semantic similarity (groupwise BMA, as
go3.compare_gene_set_pairs_batch) between each method's gene set (one per
selection) and four GWAS Catalog diseases reference gene sets plus two
curated Parkinson's panels (PanelApp and ParkinsonsUK-UCL), per cell type /
ontology / distance measure.

The gene x gene similarities come from the shared pair cache
(_go3_common.pair_similarities), so each gene pair is computed once for every
selection, method, dataset and for gene_matrix.py.

A gene without an informative annotation in the ontology (on either side) is
left out of the BMA rather than counted as 0 (_go3_common.annotated_block);
a set with no annotated gene there gives NaN.
"""

from pathlib import Path

import pandas as pd

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


# Per reference, genes carrying >=1 informative GAF annotation in each sub-ontology
def _annotated_counts(refs: dict, sym_by_go: dict, out_csv: Path) -> None:
    rows = []
    for key, genes in refs.items():
        gset = set(genes)
        for go_source, syms in sym_by_go.items():
            rows.append({"reference": key, "go_source": go_source, "n_annotated": len(gset & syms)})
        # Also record the reference's total size regardless of GO annotation
        rows.append({"reference": key, "go_source": "total", "n_annotated": len(gset)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# One similarity row per (reference, method, cell_type, ontology, distance),that's the four fors
def _compute(entries: list, refs: dict, pair_matrices: dict, distances: list, sym_by_go: dict) -> pd.DataFrame:
    rows = []
    for ref_key, ref_genes in refs.items():
        for ont in ONTOLOGIES:
            annotated = sym_by_go[GO_SOURCES[ont]]
            for dist in distances:
                # Every entry's BMA similarity is read from the cached gene x gene
                # block, as go3.compare_gene_set_pairs_batch(..., "bma") but over
                # the annotated genes only
                pm = pair_matrices[(ont, dist)]
                for e in entries:
                    block = g3c.annotated_block(pm, e["genes"], ref_genes, annotated)
                    rows.append({"reference": ref_key, "method": e["method"],
                                 "cell_type": e["cell_type"], "n_genes": len(e["genes"]),
                                 "go_source": GO_SOURCES[ont],
                                 "distance": dist, "similarity": g3c.bma(block)})
    return pd.DataFrame(rows)

def task_gwas_similarity(results_dir: Path, selections=SELECTIONS, distances=DEFAULT_DISTANCES) -> None:
    refs_dir = results_dir / "gwas_similarity" / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Fetch (or reuse the cached) GWAS Catalog gene sets for PD/AD/CAD/CKD, plus the two PD panels
    refs = {**g3c.fetch_all_gwas(refs_dir), **g3c.load_pd_panels(refs_dir)}
    print("Reference sizes:", {k: len(v) for k, v in refs.items()}, flush=True)

    # Loading the GO ontology
    counter = g3c.init_go3()
    # Record how many genes in each reference (i.e., PD/AD/CAD/CKD and the PD panels) are annotated in each sub-ontology
    sym_by_go = g3c.gaf_symbols_by_go_source()
    _annotated_counts(refs, sym_by_go, refs_dir / "annotated_counts.csv")

    # Load the gene set of every (selection, method, cell type)
    entries_by_selection = {s: _collect_entries(results_dir, s) for s in selections}
    for s, entries in entries_by_selection.items():
        print(f"[{s}] collected {len(entries)} gene sets; distances={list(distances)}", flush=True)

    # One go3 pass per (ontology, distance) over every gene of every selection,
    # computing only the gene pairs that are not in the shared cache yet
    all_genes = sorted({g for entries in entries_by_selection.values() for e in entries for g in e["genes"]})
    all_refs = sorted({g for genes in refs.values() for g in genes})
    cache_dir = g3c.pair_cache_dir(results_dir)
    pair_matrices = {(ont, dist): g3c.pair_similarities(all_genes, all_refs, ont, dist, counter, cache_dir)
                     for ont in ONTOLOGIES for dist in distances}

    for selection, entries in entries_by_selection.items():
        out_dir = output_dir(results_dir, "gwas_similarity", selection)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = _compute(entries, refs, pair_matrices, distances, sym_by_go)
        df.to_csv(out_dir / "gene_level_similarity_all.csv", index=False)
        print("Saved:", out_dir / "gene_level_similarity_all.csv", flush=True)

        plot_gwas_set_similarity(results_dir, selection)
