"""Shared GO3 / GWAS Catalog / PD panel helpers for gwas_similarity.py and gene_matrix.py.

go3 is a local Rust/PyO3 package (not on PyPI); this module only assumes it is
importable in the running environment (it already is inside scgpt_env).

go-basic.obo and goa_human.gaf are the monorepo's own reference copies
(GO3/go-basic.obo, data/goa_human.gaf at the repo root); interp_repro_mixed
expects them symlinked under data/go3_refs/ rather than duplicated, since
data/ is gitignored here.
"""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Optional

import gseapy
import numpy as np
import pandas as pd

import go3

from ..utils.progress import progress

REFS_DIR = Path(__file__).resolve().parents[2] / "data" / "go3_refs"
OBO_PATH = REFS_DIR / "go-basic.obo"
GAF_PATH = REFS_DIR / "goa_human.gaf"

GWAS_LIBRARY = "GWAS_Catalog_2025"
GWAS_TRAITS = {"parkinson": "Parkinsons Disease",
               "alzheimer": "Alzheimers Disease",
               "cad": "Coronary Artery Disease"}
ASPECT_TO_GO_SOURCE = {"P": "GO:BP", "C": "GO:CC", "F": "GO:MF"}

# Two curated Parkinson's references besides the GWAS one: the PanelApp
# "Parkinson Disease and Complex Parkinsonism" panel and the ParkinsonsUK-UCL
# GO annotations exported from QuickGO
PANELS_DIR = Path(__file__).resolve().parents[2] / "data" / "pd_panels"
PANELAPP_TSV = PANELS_DIR / "panelapp_parkinson_disease_and_complex_parkinsonism.tsv"
PARKINSONSUK_UCL_TSV = PANELS_DIR / "quickgo_parkinsonsuk_ucl.tsv"
# PanelApp status 3 or more is Green, the diagnostic-grade genes of the panel
PANELAPP_MIN_STATUS = 3


# Fetch one GWAS Catalog trait's gene set, caching it to `cache` on first use
def fetch_gwas(trait: str, cache: Path) -> list:
    if cache.exists():
        return pd.read_csv(cache)["gene"].astype(str).tolist()
    lib = gseapy.get_library(GWAS_LIBRARY)
    # Trait names in the library are free text, so match case/whitespace-insensitively
    match = [g for t, g in lib.items() if t.strip().lower() == trait.lower()]
    if not match:
        raise ValueError(f"trait {trait!r} not found in {GWAS_LIBRARY}")
    genes = sorted(dict.fromkeys(g for gs in match for g in gs))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": genes}).to_csv(cache, index=False)
    print(f"GWAS '{trait}': {len(genes)} genes -> {cache}")
    return genes

# Fetch (or reuse the cache of) every GWAS disease
def fetch_all_gwas(out_dir: Path) -> dict:
    return {key: fetch_gwas(trait, out_dir / f"{key}_genes.csv") for key, trait in GWAS_TRAITS.items()}

# Green genes of the PanelApp panel
def load_panelapp() -> list:
    df = pd.read_csv(PANELAPP_TSV, sep="\t", dtype=str)
    status = pd.to_numeric(df["GEL_Status"], errors="coerce")
    keep = (df["Entity type"] == "gene") & (status >= PANELAPP_MIN_STATUS)
    return sorted(df.loc[keep, "Gene Symbol"].dropna().unique())

# Genes with at least one ParkinsonsUK-UCL GO annotation
def load_parkinsonsuk_ucl() -> list:
    df = pd.read_csv(PARKINSONSUK_UCL_TSV, sep="\t", dtype=str, encoding="utf-8-sig")
    # A NOT qualifier says the gene lacks the function, and the ComplexPortal
    # and RNAcentral rows are complexes and RNAs, not genes
    keep = ~df["QUALIFIER"].str.startswith("NOT") & (df["GENE PRODUCT DB"] == "UniProtKB")
    return sorted(df.loc[keep, "SYMBOL"].unique())

# Load both PD panels and save them next to the GWAS gene sets, so gene_matrix.py reads them the same way
def load_pd_panels(out_dir: Path) -> dict:
    panels = {"panelapp": load_panelapp(), "parkinsonsuk_ucl": load_parkinsonsuk_ucl()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, genes in panels.items():
        pd.DataFrame({"gene": genes}).to_csv(out_dir / f"{key}_genes.csv", index=False)
    return panels

# This function gets all Gene symbols at least one GO annotation
def gaf_symbols_by_go_source() -> dict:
    gaf = pd.read_csv(GAF_PATH, sep="\t", comment="!", header=None, usecols=[2, 8],
                      names=["symbol", "aspect"], dtype=str, low_memory=False)
    return {go_source: set(gaf.loc[gaf["aspect"] == aspect, "symbol"])
            for aspect, go_source in ASPECT_TO_GO_SOURCE.items()}

# Load the GO ontology and build the annotation counter used by every go3.
# Every available core by default (GO3_THREADS overrides it)
def init_go3(threads: Optional[int] = None):
    threads = threads or int(os.environ.get("GO3_THREADS", 0)) or os.cpu_count() or 16
    go3.set_num_threads(threads)
    go3.load_go_terms(str(OBO_PATH))
    return go3.build_term_counter(go3.load_gaf(str(GAF_PATH)))


# ---------------------------------------------------------------------------
# Gene-pair similarity cache
# ---------------------------------------------------------------------------
# Gene-to-gene GO similarity only depends on the GO files, not on the dataset,
# selection or method, while the same pairs come up again and again (pooled and
# CSSI sets overlap, elbow sets sit inside z-score sets, gene_matrix compares the
# same genes as gwas_similarity). Every pair is computed once per
# (ontology, distance) and kept on disk, one matrix per file:
#   <results root>/go3_pair_cache/<ontology>__<distance>.npz
# A set-level BMA similarity is then rebuilt exactly from its gene x gene block
# (see bma), the same aggregation go3.compare_gene_set_pairs_batch does.
PAIR_GROUPWISE = "bma"
PAIR_CHUNK = 250_000


def pair_cache_dir(results_dir: Path) -> Path:
    # results_dir is results/dataset_<x>; both datasets share one cache
    return Path(results_dir).parent / "go3_pair_cache"


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_pairs(path: Path):
    if not path.exists():
        return [], [], np.zeros((0, 0))
    with np.load(path, allow_pickle=False) as npz:
        return npz["genes"].tolist(), npz["refs"].tolist(), npz["values"]


# Put `values` (rows `genes`, columns `refs`) into a NaN matrix over the union axes
def _embed(genes: list, refs: list, values: np.ndarray, all_genes: list, all_refs: list) -> np.ndarray:
    out = np.full((len(all_genes), len(all_refs)), np.nan)
    if len(genes) and len(refs):
        gi = {g: i for i, g in enumerate(all_genes)}
        ri = {r: j for j, r in enumerate(all_refs)}
        out[np.ix_([gi[g] for g in genes], [ri[r] for r in refs])] = values
    return out


class PairMatrix:
    """Gene x reference-gene similarities of one (ontology, distance)."""

    def __init__(self, genes: list, refs: list, values: np.ndarray):
        self.values = values
        self._gi = {g: i for i, g in enumerate(genes)}
        self._ri = {r: j for j, r in enumerate(refs)}

    def block(self, genes: list, refs: list) -> np.ndarray:
        return self.values[np.ix_([self._gi[g] for g in genes], [self._ri[r] for r in refs])]


def pair_similarities(genes, refs, ontology: str, distance: str, counter, cache_dir: Path) -> PairMatrix:
    """Similarity of every (gene, ref gene) pair, computing only the pairs not cached yet."""
    path = Path(cache_dir) / f"{ontology}__{distance}.npz"
    lock = path.with_suffix(".lock")
    with _locked(lock):
        old_genes, old_refs, old_values = _load_pairs(path)
    want_genes, want_refs = sorted(set(genes)), sorted(set(refs))
    all_genes = sorted(set(old_genes) | set(want_genes))
    all_refs = sorted(set(old_refs) | set(want_refs))
    values = _embed(old_genes, old_refs, old_values, all_genes, all_refs)

    gi = {g: i for i, g in enumerate(all_genes)}
    ri = {r: j for j, r in enumerate(all_refs)}
    rows = np.array([gi[g] for g in want_genes], dtype=int)
    cols = np.array([ri[r] for r in want_refs], dtype=int)
    missing = np.argwhere(np.isnan(values[np.ix_(rows, cols)]))
    n_total = rows.size * cols.size
    if len(missing):
        print(f"[go3 {ontology} {distance}] {len(missing):,} new gene pairs to compute "
              f"({n_total - len(missing):,} of {n_total:,} cached)", flush=True)
        with progress(total=len(missing), desc=f"go3 {ontology} {distance}", unit="pair",
                      unit_scale=True) as bar:
            for start in range(0, len(missing), PAIR_CHUNK):
                chunk = missing[start:start + PAIR_CHUNK]
                pairs = [(all_genes[rows[i]], all_refs[cols[j]]) for i, j in chunk]
                sims = go3.compare_gene_pairs_batch(pairs, ontology, distance, PAIR_GROUPWISE, counter)
                values[rows[chunk[:, 0]], cols[chunk[:, 1]]] = sims
                bar.update(len(chunk))
        # Another run may have added pairs meanwhile: merge the file on disk under the lock
        with _locked(lock):
            disk_genes, disk_refs, disk_values = _load_pairs(path)
            all_genes = sorted(set(all_genes) | set(disk_genes))
            all_refs = sorted(set(all_refs) | set(disk_refs))
            merged = _embed(disk_genes, disk_refs, disk_values, all_genes, all_refs)
            ours = _embed(list(gi), list(ri), values, all_genes, all_refs)
            values = np.where(np.isnan(merged), ours, merged)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, genes=np.asarray(all_genes), refs=np.asarray(all_refs), values=values)
            os.replace(tmp, path)
    return PairMatrix(all_genes, all_refs, values)


# Best-match average of a gene x gene block: every gene's best match in the
# other set, averaged over both sets' genes, exactly as go3 groupwise "bma"
def bma(block: np.ndarray) -> float:
    if block.size == 0:
        return float("nan")
    return float((block.max(axis=1).sum() + block.max(axis=0).sum()) / (block.shape[0] + block.shape[1]))
