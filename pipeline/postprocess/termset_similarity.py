"""
Term-level GO comparison, an optional alternative to the gene-level one of
gwas_similarity.py: each method's gene set (one per selection) and each
reference (the GWAS Catalog diseases and the two PD panels) are enriched with
g:Profiler in GO:BP, GO:CC and GO:MF (g:SCS < 0.05, electronic annotations
excluded), and the two sets of significant terms are compared in one level
with GO3's termset_similarity (BMA over SimRel by default; run_tasks.py's
--termset-groupwise and --termset-distances choose others).

The backgrounds are those of pipeline_unified's go_enrichment experiment:
  * a gene set against the genes its ranking scores (its all-gene score CSV,
    the only genes it can pick), inside the model's universe;
  * a reference against the model's universe, every gene of that model's
    gene_names.csv in the dataset.
Terms GO3's ontology lacks or made obsolete are left out. Where either side
has no term in an ontology the similarity is NaN, so a set's similarity to a
reference is the mean over the ontologies where both have terms.

Not part of --task postprocess and not read by the macrotable: run it with
--task termset_similarity once the gene sets exist (needs internet). Every
g:Profiler answer is cached with the query and background it came from
(termset_similarity/references/<model>/, termset_similarity/<selection>/enrichment/),
so a rerun only queries what changed.

Output: termset_similarity/<selection>/termset_similarity_all.csv, one row
per (reference, method, cell type, ontology, groupwise, distance) with the
gene-set size, its background size, the terms of each side and the similarity.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import go3
import pandas as pd
from gprofiler import GProfiler

from .. import config as C
from ..utils.progress import progress
from . import _go3_common as g3c
from .gene_sets import (ALL_SCORE_PATHS, CELL_TYPES, MODELS, SELECTIONS, load_genes, methods_for,
                        model_of, output_dir)

ONTOLOGIES = ["BP", "CC", "MF"]
GO_SOURCES = [f"GO:{o}" for o in ONTOLOGIES]
# go3's groupwise options for two term sets; simgic uses no term similarity
GROUPWISE = ["bma", "avg", "max", "hausdorff", "simgic"]
THRESHOLD = 0.05
# The public g:Profiler API fails now and then
ATTEMPTS = 3
TERM_COLUMNS = ["source", "native", "name", "p_value", "term_size", "query_size", "intersection_size"]
# Each model's genes inside a dataset folder (token -> gene), its universe
GENE_NAMES = {"scgpt": C.SCGPT_GENE_NAMES, "geneformer": C.GENEFORMER_GENE_NAMES,
              "sccello": C.SCCELLO_GENE_NAMES}


# Every gene the model has in this dataset
def _universe(data_dir: Path, model: str) -> list:
    return sorted(set(pd.read_csv(data_dir / GENE_NAMES[model])["gene"].astype(str)))


# The genes a method's ranking scores in one cell type, inside the model's universe
def _ranking_background(results_dir: Path, method: str, cell_type: str, universe: set) -> list:
    path = results_dir / ALL_SCORE_PATHS[method].format(ct=cell_type)
    if not path.exists():
        return []
    return sorted(set(pd.read_csv(path, usecols=["gene"])["gene"].astype(str)) & universe)


# g:Profiler's significant GO terms of `genes` against `background`
def _profile(genes: list, background: list) -> pd.DataFrame:
    if not genes or not background:
        return pd.DataFrame(columns=TERM_COLUMNS)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            df = GProfiler(return_dataframe=True).profile(
                organism="hsapiens", query=genes, sources=GO_SOURCES, user_threshold=THRESHOLD,
                significance_threshold_method="g_SCS", no_iea=True, domain_scope="custom",
                background=background)
            break
        except Exception as error:
            if attempt == ATTEMPTS:
                raise
            print(f"  g:Profiler failed ({error}), retrying", flush=True)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=TERM_COLUMNS)
    return df.loc[df["significant"] == True, TERM_COLUMNS]


# The cached answer of `path` if it came from the same genes and background,
# otherwise a new query, saved there with its key
def _cached_profile(path: Path, genes: list, background: list) -> pd.DataFrame:
    key = hashlib.sha1(json.dumps([sorted(genes), sorted(background), THRESHOLD]).encode()).hexdigest()
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text())["query"] == key:
        return pd.read_csv(path)
    terms = _profile(genes, background)
    path.parent.mkdir(parents=True, exist_ok=True)
    terms.to_csv(path, index=False)
    meta_path.write_text(json.dumps({"query": key, "n_genes": len(genes), "n_background": len(background)}))
    return terms


# Whether GO3's ontology has the term and has not made it obsolete
@lru_cache(maxsize=None)
def _in_go3(term: str) -> bool:
    try:
        return not go3.get_term_by_id(term).is_obsolete
    except ValueError:
        return False


# {ontology: term ids} of a g:Profiler answer, over the terms GO3 knows
def _term_sets(terms: pd.DataFrame) -> dict:
    return {source.removeprefix("GO:"): [t for t in ids if _in_go3(t)]
            for source, ids in terms.groupby("source")["native"]}


# Similarity of two term sets, NaN when either is empty
def _similarity(own: list, theirs: list, distance: str, groupwise: str, counter) -> float:
    if not own or not theirs:
        return float("nan")
    return float(go3.termset_similarity(own, theirs, distance, groupwise, counter))


def task_termset_similarity(data_dir: Path, results_dir: Path, selections=SELECTIONS,
                            groupwise=("bma",), distances=("SimRel",)) -> None:
    # The same references as gwas_similarity, cached in its folder
    refs_dir = results_dir / "gwas_similarity" / "refs"
    refs = {**g3c.fetch_all_gwas(refs_dir), **g3c.load_pd_panels(refs_dir)}
    counter = g3c.init_go3()
    # simgic ignores the term similarity, so it runs once with distance "none"
    options = [(g, d) for g in groupwise for d in (["none"] if g == "simgic" else distances)]

    # Each reference against each model's universe
    universes = {model: _universe(data_dir, model) for model in MODELS}
    ref_terms = {}
    for model, universe in universes.items():
        inside = set(universe)
        for ref_key, genes in progress(list(refs.items()), desc=f"g:Profiler references {model}", unit="ref"):
            path = results_dir / "termset_similarity" / "references" / model / f"{ref_key}.csv"
            ref_terms[(model, ref_key)] = _term_sets(_cached_profile(path, [g for g in genes if g in inside],
                                                                     universe))

    for selection in selections:
        out_dir = output_dir(results_dir, "termset_similarity", selection)
        rows = []
        pairs = [(m, c) for m in methods_for(selection) for c in CELL_TYPES]
        for method, cell_type in progress(pairs, desc=f"termset {selection}", unit="set"):
            genes = load_genes(results_dir, method, selection, cell_type)
            if not genes:
                continue
            model = model_of(method)
            inside = set(universes[model])
            # Each gene set against the genes its own ranking scores
            background = _ranking_background(results_dir, method, cell_type, inside)
            path = out_dir / "enrichment" / method / f"{cell_type}.csv"
            own_terms = _term_sets(_cached_profile(path, [g for g in genes if g in inside], background))
            for ref_key in refs:
                for ont in ONTOLOGIES:
                    own, theirs = own_terms.get(ont, []), ref_terms[(model, ref_key)].get(ont, [])
                    for g, d in options:
                        rows.append({"reference": ref_key, "method": method, "cell_type": cell_type,
                                     "n_genes": len(genes), "n_background": len(background),
                                     "go_source": f"GO:{ont}", "n_terms": len(own), "n_ref_terms": len(theirs),
                                     "groupwise": g, "distance": d,
                                     "similarity": _similarity(own, theirs, "SimRel" if d == "none" else d,
                                                               g, counter)})
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "termset_similarity_all.csv", index=False)
        print("Saved:", out_dir / "termset_similarity_all.csv", flush=True)
