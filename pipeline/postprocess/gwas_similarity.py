"""
Gene-level GO3 semantic similarity (go3.compare_gene_set_pairs_batch, BMA)
between the top-30 genes of each method and three GWAS Catalog diseases
reference gene sets plus two curated Parkinson's panels (PanelApp and
ParkinsonsUK-UCL), per cell type / ontology / distance measure.
"""

from pathlib import Path

import pandas as pd

import go3

from . import _go3_common as g3c
from .enrichment import CELL_TYPES, METHODS

TOP_N = 30
ONTOLOGIES = ["BP", "CC", "MF"]
GO_SOURCES = {"BP": "GO:BP", "CC": "GO:CC", "MF": "GO:MF"}
DEFAULT_DISTANCES = ["lin", "SimRel", "wang"]


# top30_genes.csv is already the top-30, but re-sorting/truncating here keeps
# this function correct even if that assumption ever changes upstream
def _top_genes(path: Path) -> list:
    if not path.exists():
        return None
    df = pd.read_csv(path).sort_values("score", ascending=False)
    return df["gene"].astype(str).head(TOP_N).tolist()


# One entry per (method, cell type) with its top-30 genes, skipping missing pairs
def _collect_entries(results_dir: Path) -> list:
    entries = []
    for method in METHODS:
        for cell_type in CELL_TYPES:
            genes = _top_genes(results_dir / method / cell_type / "top30_genes.csv")
            if genes:
                entries.append({"method": method, "cell_type": cell_type, "genes": genes})
            else:
                print(f"  MISSING {results_dir / method / cell_type / 'top30_genes.csv'}")
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
                                 "cell_type": e["cell_type"], "go_source": GO_SOURCES[ont],
                                 "distance": dist, "similarity": float(s)})
            print(f"[{ref_key}] {GO_SOURCES[ont]} done")
    return pd.DataFrame(rows)

def task_gwas_similarity(results_dir: Path) -> None:
    out_dir = results_dir / "gwas_similarity"
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Fetch (or reuse the cached) GWAS Catalog gene sets for PD/AD/CAD, plus the two PD panels
    refs = {**g3c.fetch_all_gwas(refs_dir), **g3c.load_pd_panels(refs_dir)}
    print("Reference sizes:", {k: len(v) for k, v in refs.items()})

    # Load the top-30 genes of every (method, cell type)
    entries = _collect_entries(results_dir)
    print(f"Collected {len(entries)} gene sets; distances={DEFAULT_DISTANCES}")

    # Loading the GO ontology
    counter = g3c.init_go3()
    # Record how many genes in each reference (i.e., PD/AD/CAD and the PD panels) are annotated in each sub-ontology
    _annotated_counts(refs, refs_dir / "annotated_counts.csv")

    df = _compute(entries, refs, counter, DEFAULT_DISTANCES)
    df.to_csv(out_dir / "gene_level_similarity_all.csv", index=False)
    print("Saved:", out_dir / "gene_level_similarity_all.csv")
