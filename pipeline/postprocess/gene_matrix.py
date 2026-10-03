"""Gene-to-gene GO semantic-similarity matrices between each method's gene set
(one per selection) and the Parkinson's references (GWAS, PanelApp,
ParkinsonsUK-UCL).

For every (selection x cell_type x disease x distance x ontology) builds a
matrix of shape (sum of method gene-set sizes) x (n_disease_genes), each cell
the gene-pair similarity from go3.compare_gene_pairs_batch (groupwise BMA).
Lin/SimRel are GO term-similarity measures, so the cell values are
*similarities* (higher = closer).

Adapted from code/comparison_geneset_ref/compute_genematrix_disease.py, dropped
to a single dataset and the methods this package actually produces (no
RF/Lasso baselines here). Reuses the reference gene lists cached by
gwas_similarity.py.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

from ..utils.progress import progress
from . import _go3_common as g3c
from .gene_sets import CELL_TYPES, SELECTIONS, load_genes, methods_for, output_dir
from .gwas_similarity import DEFAULT_DISTANCES
from .plots import plot_gwas_similarity

ONTOLOGIES = ["BP", "CC", "MF"]
# Diseases to compare against: the PD GWAS set and the two PD panels
DEFAULT_DISEASES = ["parkinson", "panelapp", "parkinsonsuk_ucl"]
PCTS = [10, 25, 75, 90, 95]


# One entry per (cell_type, method) with its gene set, skipping missing pairs
def _collect_entries(results_dir: Path, selection: str) -> list:
    entries = []
    for cell_type in CELL_TYPES:
        for method in methods_for(selection):
            genes = load_genes(results_dir, method, selection, cell_type)
            if genes:
                entries.append({"cell_type": cell_type, "method": method, "genes": genes})
    return entries


# Load the GWAS reference gene lists cached by gwas_similarity.py
def _load_refs(refs_dir: Path, diseases: list) -> dict:
    return {d: pd.read_csv(refs_dir / f"{d}_genes.csv")["gene"].astype(str).tolist() for d in diseases}


# (len(genes) x len(ref_genes)) matrix of similarities, read from the shared
# gene-pair cache (_go3_common.pair_similarities, groupwise BMA per gene pair)
def _build_matrix(genes: list, ref_genes: list, pm: g3c.PairMatrix) -> pd.DataFrame:
    return pd.DataFrame(pm.block(genes, ref_genes), index=genes, columns=ref_genes, dtype=float)


# Distribution stats of one block of matrix cells (`values`) plus, per gene,
# how close its single best-matching disease gene is (`bestmatch`)
def _block_stats(values: np.ndarray, bestmatch: np.ndarray) -> dict:
    out = {"mean": float(np.mean(values)), "median": float(np.median(values)),
           "std": float(np.std(values)), "max": float(np.max(values)),
           "skew": float(scipy_skew(values)) if np.std(values) > 0 else 0.0}
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(values, p))
    out["bestmatch_mean"] = float(np.mean(bestmatch))
    out["bestmatch_median"] = float(np.median(bestmatch))
    out["bestmatch_max"] = float(np.max(bestmatch))
    return out


# Per-method-block + all-methods statistics, computed both over all genes and
# over GO-annotated method genes only (zeros from unannotated genes otherwise
# depress the means, since they can never match a disease gene)
def _summarize(matrix: pd.DataFrame, ann_set: set, meta: dict) -> list:
    rows = []
    by_method = matrix.groupby(level="method", sort=False)
    groups = [(m, by_method.get_group(m)) for m in matrix.index.unique("method")]
    groups.append(("all_methods", matrix))
    for method, block in groups:
        genes = block.index.get_level_values("gene")
        ann_mask = genes.isin(ann_set)
        for scope, mask in [("all", np.ones(len(genes), bool)), ("annotated", ann_mask)]:
            sub = block.to_numpy()[mask]
            if sub.size == 0:
                continue
            # Row maxima = each gene's single closest match among the disease genes
            stats = _block_stats(sub.ravel(), sub.max(axis=1))
            rows.append({**meta, "method": method, "scope": scope,
                         "n_genes": int(mask.sum()), "n_annotated": int(ann_mask.sum()),
                         "n_ref_genes": matrix.shape[1], **stats})
    return rows


def _gene_matrix_for_selection(results_dir: Path, selection: str, entries: list, refs: dict,
                               pair_matrices: dict, distances: list, ann: dict) -> None:
    out_dir = output_dir(results_dir, "gene_matrix", selection)
    (out_dir / "matrices").mkdir(parents=True, exist_ok=True)
    print(f"[{selection}] entries:", len(entries), "| diseases:", {d: len(g) for d, g in refs.items()},
          flush=True)

    # Group entries variable by cell_type so every method's matrix for that cell_type
    # can be stacked into one file below
    groups: dict = {}
    for e in entries:
        groups.setdefault(e["cell_type"], []).append(e)

    # Iterate though references, ontologies and distances
    summary_rows = []
    for ont, distance in progress(list(product(ONTOLOGIES, distances)), desc=f"gene_matrix {selection}",
                                  unit="matrix set"):
        # One cached gene-pair matrix per (ontology, distance) covers every cell_type/disease pair below
        pm = pair_matrices[(ont, distance)]
        for cell_type, ents in groups.items():
            for disease, ref_genes in refs.items():
                # Stack each method's (n_genes x n_ref_genes) block into one matrix,
                # tagged by method in the row index
                blocks = []
                for e in ents:
                    # Create the matrix of our method genes vs the GWAS disease reference gene-set
                    mat = _build_matrix(e["genes"], ref_genes, pm)
                    mat.index = pd.MultiIndex.from_product([[e["method"]], mat.index], names=["method", "gene"])
                    blocks.append(mat)
                full = pd.concat(blocks)
                out_csv = out_dir / "matrices" / cell_type / f"{disease}__{distance}__{ont}.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                full.to_csv(out_csv)
                meta = {"cell_type": cell_type, "disease": disease, "distance": distance, "ontology": ont}
                summary_rows += _summarize(full, ann[ont], meta)

    # One row per (cell_type, method, gene, ontology), whether that gene has
    # any GO annotation there, independent of any disease/distance/matrix above
    ann_rows = [{"cell_type": e["cell_type"], "method": e["method"], "gene": g,
                 "ontology": ont, "annotated": g in ann[ont]}
                for ont in ONTOLOGIES for e in entries for g in e["genes"]]
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_statistics.csv", index=False)
    pd.DataFrame(ann_rows).to_csv(out_dir / "method_gene_annotation.csv", index=False)
    print("Saved:", out_dir / "summary_statistics.csv")

    plot_gwas_similarity(results_dir, selection)


def task_gene_matrix(results_dir: Path, selections=SELECTIONS, distances=DEFAULT_DISTANCES) -> None:
    refs = _load_refs(results_dir / "gwas_similarity" / "refs", DEFAULT_DISEASES)

    # Loading the GO ontology
    counter = g3c.init_go3()
    sym_by_go = g3c.gaf_symbols_by_go_source()
    ann = {ont: sym_by_go[f"GO:{ont}"] for ont in ONTOLOGIES}

    # Gene pairs of every selection at once; after gwas_similarity.py they are
    # all in the shared cache already, so nothing is recomputed here
    entries_by_selection = {s: _collect_entries(results_dir, s) for s in selections}
    all_genes = sorted({g for entries in entries_by_selection.values() for e in entries for g in e["genes"]})
    all_refs = sorted({g for genes in refs.values() for g in genes})
    cache_dir = g3c.pair_cache_dir(results_dir)
    pair_matrices = {(ont, dist): g3c.pair_similarities(all_genes, all_refs, ont, dist, counter, cache_dir)
                     for ont in ONTOLOGIES for dist in distances}

    for selection, entries in entries_by_selection.items():
        _gene_matrix_for_selection(results_dir, selection, entries, refs, pair_matrices, distances, ann)
