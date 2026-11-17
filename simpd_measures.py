#!/usr/bin/env python
"""Compare measures of PD similarity (SimPD variants) by their p-values against random gene sets.

Every measure keeps SimPD's formula, mean of the PD references (PD GWAS,
PanelApp, ParkinsonsUK-UCL) minus mean of the control diseases (Alzheimer,
kidney, CAD), each reference's similarity averaged over the measure's
ontologies; only how a gene set's similarity to a reference is computed
changes (docs/go3_explicado.md, plan docs/plan_simpd_lora_2026-11.md):

  measure        term       gene-gene  set-reference  ontologies
  M0_simrel_bma  SimRel     BMA        BMA            BP CC MF   (the current SimPD)
  M1_simrel_median          BMA        median         BP CC MF   (JA's best combination)
  M2_simrel_mean            BMA        mean
  M3_simrel_max             BMA        max
  M3_simrel_hausdorff       BMA        Hausdorff (worst best match)
  M4_wang_bma / M4_wang_median   Wang, BMA, BMA / median
  M5_lin_bma / M5_lin_median     Lin, BMA, BMA / median
  M6_simrel_bma_bp               as M0, BP only
  M7_profile     SimRel, no gene level: the GO-term profiles of the set and the
                 reference (each term weighted by its number of genes), weighted
                 BMA (go3.compare_gene_set_profiles, rebuilt exactly)
  M8_resnik_bma  Resnik / max of the matrix (per ontology), BMA, BMA
  M9_simrel_max  SimRel, gene-gene max (the best term pair), BMA
  M10_resnik_max Resnik normalised, gene-gene max, BMA

Gene x gene similarities come from the shared GO3 pair cache
(results/scgpt/go3_pair_cache, _go3_common.pair_similarities); M7 uses a
term x term SimRel matrix instead (<ontology>__SimRel__profile_terms.npz in
the same folder, ~3 min to build). The pairs a measure lacks are computed first (--compute-pairs only does that, for a CPU
node: tens of millions of pairs per ontology for a new distance).

Gene sets: the pooled |IG| rankings of --forms (default mlp, lora, step2_5,
written as self-supervised in the CSVs)
x top30 / z-score / elbow x 4 cell types x --datasets. Each is compared with
two nulls of --draws random sets drawn exactly as simpd_baseline.py does
(same seeds, so M0 reproduces its p-values): uniform (same size, any gene of
the cell type's universe) and expr (same number of genes per detection-rate
decile, the main null). p = (1 + #draws >= observed) / (1 + draws), one-sided;
q = Benjamini-Hochberg over the rows of one measure.

Planted genes (--spike-k, --spike-sets): random 30-gene sets of the dataset's
universe with k genes planted from one half (H1, seed 42) of the PD
references' genes, scored against the references without H1 (a planted gene
never matches itself), and the same planting AD GWAS genes (the specificity
control: SimPD should not rise). Per measure and k: the fraction of sets with
p < 0.05 against uniform sets scored with the same references (k = 0 is the
type I error, ~0.05 by construction).

Outputs in <results-root>/simpd_measures/:
  observed.csv    measure, dataset, form, cell_type, selection, n_genes,
                  SimPDGWAS, SimPDGE, SimPDGO, SimAD, SimKidney, SimCAD (the
                  macrotable's names), SimPD, p_uniform, p_expr, q_expr
  aggregate.csv   mean over the cell types against the null of that mean
  spike.csv       measure, dataset, planted (PD/AD), k, n_sets, frac_p05 and
                  its 95 % Wilson interval, median SimPD
  summary.csv     per measure: fraction of gene sets with p_expr < 0.05 and
                  median p_expr per form, sensitivity (PD, k = 5),
                  specificity control (AD, k = 5), type I (k = 0)
  comparison.csv  per measure, planted genes pooled over the datasets: type I,
                  PD k = 2 / 5 / 10, AD k = 5 / 10, Wilson intervals, and
                  whether it beats M0 (type I <= 0.075, AD k = 5 not above
                  M0's, PD k = 5 interval above M0's)
  mlp_vs_lora.csv per measure: zero-shot+MLP vs LoRA paired over dataset x
                  cell type x cut-off (log10 p_expr and SimPD differences,
                  positive = LoRA better, two-sided Wilcoxon)

  python simpd_measures.py --results-root results/scgpt --datasets A B
  python simpd_measures.py --measures M8_resnik_bma M9_simrel_max --compute-pairs   # CPU node

On another machine, the same genes and references: --write-pairs-file
pairs.json here, then --compute-pairs --pairs-file pairs.json --cache-dir
<dir> there (only go3, the GO files of data/go3_refs and this file needed),
and copy <dir>/*.npz back into results/scgpt/go3_pair_cache.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import go3
import numpy as np
import pandas as pd

from pipeline.postprocess import _go3_common as g3c
from pipeline.postprocess import gene_sets as gs
from pipeline.utils.progress import progress
from pipeline.macrotable.columns.similarity import REFERENCES
from simpd_baseline import N_BINS, SELECTION_DIRS, SIGN, SEED, _detection_bins, _universe

ONTOLOGIES = ["BP", "CC", "MF"]
FORMS = ["mlp", "lora", "step2_5"]
# Name of each form in the output CSVs (folders and tasks keep step2_5)
FORM_LABELS = {"step2_5": "self-supervised"}
# Column of the similarity to each reference in the CSVs: the macrotable's names (SimPDGWAS, ..., SimCAD)
REF_COLUMNS = {ref: f"Sim{label}" for label, ref in REFERENCES.items()}
# measure -> term distance, gene-level groupwise, set-level aggregation, ontologies, Resnik-style normalisation
MEASURES = {
    "M0_simrel_bma": ("SimRel", "bma", "bma", ONTOLOGIES, False),
    "M1_simrel_median": ("SimRel", "bma", "median", ONTOLOGIES, False),
    "M2_simrel_mean": ("SimRel", "bma", "mean", ONTOLOGIES, False),
    "M3_simrel_max": ("SimRel", "bma", "max", ONTOLOGIES, False),
    "M3_simrel_hausdorff": ("SimRel", "bma", "hausdorff", ONTOLOGIES, False),
    "M4_wang_bma": ("wang", "bma", "bma", ONTOLOGIES, False),
    "M4_wang_median": ("wang", "bma", "median", ONTOLOGIES, False),
    "M5_lin_bma": ("lin", "bma", "bma", ONTOLOGIES, False),
    "M5_lin_median": ("lin", "bma", "median", ONTOLOGIES, False),
    "M6_simrel_bma_bp": ("SimRel", "bma", "bma", ["BP"], False),
    "M7_profile": ("SimRel", "profile", "profile", ONTOLOGIES, False),
    "M8_resnik_bma": ("resnik", "bma", "bma", ONTOLOGIES, True),
    "M9_simrel_max": ("SimRel", "max", "bma", ONTOLOGIES, False),
    "M10_resnik_max": ("resnik", "max", "bma", ONTOLOGIES, True),
}
# Every measure (their pairs for the whole universe are in the cache since 2026-11-01)
DEFAULT_MEASURES = list(MEASURES)
REF_ORDER = list(SIGN)
PD_REFS = ["parkinson", "panelapp", "parkinsonsuk_ucl"]
# Gathered values per scoring chunk (draws x genes x reference genes), to bound memory
CHUNK_VALUES = 30_000_000
NAMESPACES = {"BP": "biological_process", "CC": "cellular_component", "MF": "molecular_function"}
# Term pairs per go3.batch_similarity call (M7's term x term matrix)
TERM_CHUNK = 2_000_000


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _aggregate(block: np.ndarray, how: str) -> np.ndarray:
    """Set-level similarity of each draw: block is (draws, set genes, reference genes)."""
    if how == "bma":
        return (block.max(axis=2).sum(axis=1) + block.max(axis=1).sum(axis=1)) / (block.shape[1] + block.shape[2])
    if how == "median":
        return np.median(block.reshape(block.shape[0], -1), axis=1)
    if how == "mean":
        return block.mean(axis=(1, 2))
    if how == "max":
        return block.max(axis=(1, 2))
    if how == "hausdorff":
        return np.minimum(block.max(axis=2).min(axis=1), block.max(axis=1).min(axis=1))
    raise ValueError(how)


class Scorer:
    """SimPD of sets of universe genes (index arrays) under the measures of one pair-cache configuration."""

    def __init__(self, universe: list, refs: dict, mats: dict, scale: dict):
        # Per ontology and reference, the universe x reference-gene block (float32 to save memory)
        self.blocks = {}
        for o, pm in mats.items():
            rows = np.array([pm._gi[g] for g in universe])
            for r, genes in refs.items():
                cols = [pm._ri[g] for g in genes]
                self.blocks[(o, r)] = (pm.values[np.ix_(rows, cols)] / scale[o]).astype(np.float32)

    def per_ref(self, picks: np.ndarray, how: str, ontologies: list) -> dict:
        """{reference: (draws,) similarity averaged over the ontologies} for picks (draws, set genes)."""
        out = {}
        for r in REF_ORDER:
            sims = []
            for o in ontologies:
                block = self.blocks[(o, r)]
                step = max(1, CHUNK_VALUES // (picks.shape[1] * block.shape[1]))
                sims.append(np.concatenate([_aggregate(block[picks[s:s + step]], how)
                                            for s in range(0, picks.shape[0], step)]))
            out[r] = np.mean(sims, axis=0)
        return out


# ---------------------------------------------------------------------------
# M7: GO-term profiles (go3.compare_gene_set_profiles)
# ---------------------------------------------------------------------------
# go3 builds each gene set's profile from its genes' GO terms of one ontology
# (as annotated, not propagated), weighting each term by how many of the genes
# have it, and compares two profiles with a weighted BMA:
#   sim = (sum_t w_t max_c S(t, c) + sum_c u_c max_t S(t, c)) / (sum_t w_t + sum_c u_c)
# over the set's terms t (weights w) and the reference's terms c (weights u).
# Both sums split by gene: the first is the sum over the set's genes of the sum
# of their terms' row maxima, the second needs only each gene's column maxima.
# So, with the SimRel matrix of every gene term x every reference term, a set's
# value is a few array sums (it reproduces go3 to ~1e-7; go3 itself takes ~1 s
# per set and reference). Genes go3 does not know give no terms (go3 raises).

_GENE2GO = {}


def _gene2go() -> dict:
    """Gene symbol -> its GO terms, exactly as go3 maps them (its load_gaf filters, keyed by the GAF symbol)."""
    if not _GENE2GO:
        ann = go3.load_gaf(str(g3c.GAF_PATH))
        gaf = pd.read_csv(g3c.GAF_PATH, sep="\t", comment="!", header=None, usecols=[1, 2, 4], dtype=str)
        uid, sym, term = (gaf[c].to_numpy() for c in (1, 2, 4))
        # go3 keeps the GAF lines in order and drops some: each annotation is the next line with its id and term
        j = 0
        for a in ann:
            while uid[j] != a.db_object_id or term[j] != a.go_term:
                j += 1
            _GENE2GO.setdefault(sym[j], set()).add(a.go_term)
            j += 1
    return _GENE2GO


class ProfileTerms:
    """One ontology's GO terms of the genes and of the reference genes, and their SimRel term x term matrix."""

    def __init__(self, genes: list, refs: list, ontology: str, counter, cache_dir: Path):
        gene2go, namespace = _gene2go(), {}

        def in_ontology(term):
            if term not in namespace:
                try:
                    namespace[term] = go3.get_term_by_id(term).namespace
                except ValueError:
                    namespace[term] = None
            return namespace[term] == NAMESPACES[ontology]

        self.gene_terms = {g: sorted(t for t in gene2go.get(g, ()) if in_ontology(t)) for g in set(genes) | set(refs)}
        rows = sorted({t for g in genes for t in self.gene_terms[g]})
        cols = sorted({t for g in refs for t in self.gene_terms[g]})
        path = Path(cache_dir) / f"{ontology}__SimRel__profile_terms.npz"
        self.values = None
        if path.exists():
            z = np.load(path)
            if z["rows"].tolist() == rows and z["cols"].tolist() == cols:
                self.values = z["values"]
        if self.values is None:
            start = time.time()
            self.values = np.empty((len(rows), len(cols)), dtype=np.float32)
            step = max(1, TERM_CHUNK // max(1, len(cols)))
            with progress(total=len(rows), desc=f"go3 {ontology} term pairs", unit="term") as bar:
                for s in range(0, len(rows), step):
                    block = rows[s:s + step]
                    v = go3.batch_similarity([a for a in block for _ in cols], cols * len(block), "SimRel", counter)
                    self.values[s:s + len(block)] = np.asarray(v, dtype=np.float32).reshape(len(block), len(cols))
                    bar.update(len(block))
            np.savez(path, rows=np.array(rows), cols=np.array(cols), values=self.values)
            print(f"  {ontology}: {len(rows)} x {len(cols)} term pairs in {time.time() - start:.0f} s -> {path}",
                  flush=True)
        self.ri = {t: i for i, t in enumerate(rows)}
        self.ci = {t: i for i, t in enumerate(cols)}


class ProfileScorer:
    """SimPD of sets of universe genes under M7 (same interface as Scorer)."""

    def __init__(self, universe: list, refs: dict, mats: dict):
        self.parts = {}
        for o, pt in mats.items():
            rows_of = [np.array([pt.ri[t] for t in pt.gene_terms.get(g, [])], dtype=int) for g in universe]
            n_terms = np.array([r.size for r in rows_of], dtype=np.float64)
            for r, genes in refs.items():
                u = Counter(c for g in set(genes) for c in pt.gene_terms.get(g, []))
                terms = sorted(u)
                weights = np.array([u[c] for c in terms], dtype=np.float64)
                sub = pt.values[:, [pt.ci[c] for c in terms]]
                row_max = sub.max(axis=1) if terms else np.zeros(sub.shape[0], dtype=np.float32)
                gene_row = np.array([row_max[i].sum() for i in rows_of], dtype=np.float64)
                gene_col = np.zeros((len(universe), len(terms)), dtype=np.float32)
                for k, i in enumerate(rows_of):
                    if i.size:
                        gene_col[k] = sub[i].max(axis=0)
                self.parts[(o, r)] = (gene_row, n_terms, gene_col, weights)

    def per_ref(self, picks: np.ndarray, how: str, ontologies: list) -> dict:
        out = {}
        for r in REF_ORDER:
            sims = []
            for o in ontologies:
                gene_row, n_terms, gene_col, u = self.parts[(o, r)]
                step = max(1, CHUNK_VALUES // (picks.shape[1] * max(1, u.size)))
                vals = []
                for s in range(0, picks.shape[0], step):
                    p = picks[s:s + step]
                    w = n_terms[p].sum(axis=1)
                    num = gene_row[p].sum(axis=1) + gene_col[p].max(axis=1) @ u
                    # go3 gives 0 when either profile is empty
                    vals.append(np.where((w > 0) & (u.sum() > 0), num / np.maximum(w + u.sum(), 1e-12), 0.0))
                sims.append(np.concatenate(vals))
            out[r] = np.mean(sims, axis=0)
        return out


def _make_scorer(universe: list, refs: dict, mats: dict, scale: dict):
    if isinstance(next(iter(mats.values())), ProfileTerms):
        return ProfileScorer(universe, refs, mats)
    return Scorer(universe, refs, mats, scale)


def _simpd(per_ref: dict) -> np.ndarray:
    return sum(SIGN[r] * v for r, v in per_ref.items())


def _p(obs: float, null: np.ndarray) -> float:
    return float((1 + (null >= obs).sum()) / (1 + null.size))


def _bh(p: pd.Series) -> pd.Series:
    valid = p.dropna()
    n = len(valid)
    if n == 0:
        return p.copy()
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order] * n / np.arange(1, n + 1)
    q = np.empty(n)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return pd.Series(q, index=valid.index).reindex(p.index)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Random draws (exactly as simpd_baseline.py, so M0 reproduces its p-values)
# ---------------------------------------------------------------------------

def _uniform_picks(dataset: str, ct_index: int, n_universe: int, n_genes: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng([SEED, ord(dataset.upper()), ct_index, 0, n_genes])
    return np.stack([rng.choice(n_universe, n_genes, replace=False) for _ in range(draws)])


def _expr_picks(dataset: str, ct_index: int, members: list, counts: np.ndarray, draws: int) -> np.ndarray:
    rng = np.random.default_rng([SEED, ord(dataset.upper()), ct_index, 1, *counts.tolist()])
    return np.stack([np.concatenate([rng.choice(members[b], c, replace=False) for b, c in enumerate(counts) if c])
                     for _ in range(draws)])


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _refs(results_dir: Path) -> dict:
    refs_dir = results_dir / "gwas_similarity" / "refs"
    refs = {**g3c.fetch_all_gwas(refs_dir), **g3c.load_pd_panels(refs_dir)}
    return {r: refs[r] for r in REF_ORDER}


def _gene_sets(results_dir: Path, forms: list) -> list:
    """(form, selection, cell type, genes) of every pooled |IG| list of the forms."""
    out = []
    for form in forms:
        method = f"ig_{form}_scgpt"
        for selection in SELECTION_DIRS:
            for ct in gs.CELL_TYPES:
                genes = gs.load_genes(results_dir, method, selection, ct)
                if genes:
                    out.append((form, selection, ct, genes))
    return out


def _configs(measures: list) -> dict:
    """{(distance, groupwise): [measures]}: the measures sharing one pair cache."""
    out = {}
    for m in measures:
        dist, gene, _, _, _ = MEASURES[m]
        out.setdefault((dist, gene), []).append(m)
    return out


def _compute_pairs(measures: list, genes: list, refs: list, counter, cache_dir: Path) -> None:
    for (dist, gene), ms in _configs(measures).items():
        onts = sorted({o for m in ms for o in MEASURES[m][3]})
        start = time.time()
        _load_mats(dist, gene, onts, genes, refs, counter, cache_dir)
        print(f"pairs of {dist} / {gene} ({', '.join(ms)}) ready in {time.time() - start:.0f} s", flush=True)


def _load_mats(dist: str, gene: str, ontologies: list, genes: list, refs: list, counter, cache_dir: Path) -> dict:
    if gene == "profile":
        return {o: ProfileTerms(genes, refs, o, counter, cache_dir) for o in ontologies}
    return {o: g3c.pair_similarities(genes, refs, o, dist, counter, cache_dir, groupwise=gene) for o in ontologies}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    out_dir = args.out_dir or args.results_root / "simpd_measures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.results_root / "go3_pair_cache"
    if args.compute_pairs and args.pairs_file:
        # Fixed genes and references from another machine: no results needed here
        pairs = json.load(open(args.pairs_file))
        counter = g3c.init_go3(args.threads)
        _compute_pairs(args.measures, pairs["genes"], pairs["refs"], counter, cache_dir)
        return
    data = {}
    for ds in args.datasets:
        results_dir = args.results_root / f"dataset_{ds.lower()}"
        universes = {ct: _universe(results_dir, ct) for ct in gs.CELL_TYPES}
        data[ds] = {"dir": results_dir, "refs": _refs(results_dir), "universes": universes,
                    "sets": _gene_sets(results_dir, args.forms)}
        print(f"{ds}: {len(data[ds]['sets'])} gene sets, universes "
              f"{ {ct: len(u) for ct, u in universes.items()} }", flush=True)
    all_genes = sorted(set().union(*[set().union(*d["universes"].values()) for d in data.values()]))
    all_refs = sorted({g for d in data.values() for genes in d["refs"].values() for g in genes})
    if args.write_pairs_file:
        json.dump({"genes": all_genes, "refs": all_refs}, open(args.write_pairs_file, "w"))
        print(f"{len(all_genes)} genes x {len(all_refs)} reference genes -> {args.write_pairs_file}", flush=True)
        return
    counter = g3c.init_go3(args.threads)

    if args.compute_pairs:
        _compute_pairs(args.measures, all_genes, all_refs, counter, cache_dir)
        return

    rows, agg_rows, spike_rows = [], [], []
    for (dist, gene), ms in _configs(args.measures).items():
        onts = sorted({o for m in ms for o in MEASURES[m][3]}, key=ONTOLOGIES.index)
        mats = _load_mats(dist, gene, onts, all_genes, all_refs, counter, cache_dir)
        # Resnik is unbounded: divide by the largest gene-pair value of each ontology
        normalise = any(MEASURES[m][4] for m in ms)
        scale = {o: float(np.nanmax(mats[o].values)) if normalise else 1.0 for o in onts}
        print(f"\n# {dist} / {gene}: {', '.join(ms)}", flush=True)
        for ds, d in data.items():
            null_means = {}
            for i, ct in enumerate(gs.CELL_TYPES):
                universe = d["universes"][ct]
                index = {g: k for k, g in enumerate(universe)}
                scorer = _make_scorer(universe, d["refs"], mats, scale)
                bins = _detection_bins(d["dir"], ct, universe)
                members = [np.where(bins == b)[0] for b in range(N_BINS)]
                uniform_cache = {}
                sets = [s for s in d["sets"] if s[2] == ct]
                start = time.time()
                for form, selection, _, genes in sets:
                    pick = np.array([index[g] for g in genes if g in index])
                    n = pick.size
                    if n not in uniform_cache:
                        uniform_cache[n] = _uniform_picks(ds, i, len(universe), n, args.draws)
                    counts = np.bincount(bins[pick], minlength=N_BINS)
                    expr = _expr_picks(ds, i, members, counts, args.draws)
                    for m in ms:
                        _, _, how, m_onts, _ = MEASURES[m]
                        obs_ref = scorer.per_ref(pick[None, :], how, m_onts)
                        obs = float(_simpd(obs_ref)[0])
                        nulls = {"uniform": _simpd(scorer.per_ref(uniform_cache[n], how, m_onts)),
                                 "expr": _simpd(scorer.per_ref(expr, how, m_onts))}
                        null_means[(m, form, selection, ct)] = (obs, nulls)
                        rows.append({"measure": m, "dataset": ds, "form": form, "cell_type": ct,
                                     "selection": selection, "n_genes": n,
                                     **{REF_COLUMNS[r]: float(v[0]) for r, v in obs_ref.items()},
                                     "SimPD": obs,
                                     "p_uniform": _p(obs, nulls["uniform"]), "p_expr": _p(obs, nulls["expr"])})
                print(f"  {ds} {ct}: {len(sets)} sets x {len(ms)} measures in {time.time() - start:.0f} s",
                      flush=True)
            # Mean over the cell types against the null of that mean (draw i of every cell type averaged)
            for m in ms:
                for form in args.forms:
                    for selection in SELECTION_DIRS:
                        cts = [ct for ct in gs.CELL_TYPES if (m, form, selection, ct) in null_means]
                        if len(cts) != len(gs.CELL_TYPES):
                            continue
                        obs = np.mean([null_means[(m, form, selection, ct)][0] for ct in cts])
                        row = {"measure": m, "dataset": ds, "form": form, "selection": selection, "SimPD": obs}
                        for name in ("uniform", "expr"):
                            v = np.mean([null_means[(m, form, selection, ct)][1][name] for ct in cts], axis=0)
                            row[f"p_{name}"] = _p(obs, v)
                        agg_rows.append(row)
            if args.spike_sets:
                spike_rows += _spike(ds, d, mats, scale, ms, args)
        del mats

    observed = pd.DataFrame(rows)
    observed["form"] = observed["form"].replace(FORM_LABELS)
    observed["q_expr"] = observed.groupby("measure")["p_expr"].transform(_bh)
    observed.to_csv(out_dir / "observed.csv", index=False)
    aggregate = pd.DataFrame(agg_rows)
    aggregate["form"] = aggregate["form"].replace(FORM_LABELS)
    aggregate.to_csv(out_dir / "aggregate.csv", index=False)
    spike = pd.DataFrame(spike_rows)
    if len(spike):
        spike.to_csv(out_dir / "spike.csv", index=False)
    summary = _summary(observed, spike, [FORM_LABELS.get(f, f) for f in args.forms])
    summary.to_csv(out_dir / "summary.csv", index=False)
    if len(spike):
        comparison = _comparison(spike)
        comparison.to_csv(out_dir / "comparison.csv", index=False)
        print(f"\n# Planted genes, A and B pooled\n{comparison.round(3).to_string(index=False)}", flush=True)
    if {"mlp", "lora"} <= set(args.forms):
        pairs = _mlp_vs_lora(observed)
        pairs.to_csv(out_dir / "mlp_vs_lora.csv", index=False)
        print(f"\n# zero-shot+MLP vs LoRA\n{pairs.round(3).to_string(index=False)}", flush=True)
    json.dump({"measures": args.measures, "datasets": args.datasets, "forms": args.forms, "draws": args.draws,
               "spike_k": args.spike_k, "spike_sets": args.spike_sets, "seed": SEED},
              open(out_dir / "manifest.json", "w"), indent=1)
    print(f"\n# Summary\n{summary.round(3).to_string(index=False)}\n\noutputs in {out_dir}", flush=True)


def _spike(ds: str, d: dict, mats: dict, scale: dict, measures: list, args) -> list:
    """Planted PD / AD genes in random 30-gene sets of the dataset's universe."""
    universe = sorted(set().union(*d["universes"].values()))
    index = {g: k for k, g in enumerate(universe)}
    rows = []
    pools = {"PD": sorted({g for r in PD_REFS for g in d["refs"][r]} & set(universe)),
             "AD": sorted(set(d["refs"]["alzheimer"]) & set(universe))}
    for s, (planted, pool) in enumerate(pools.items()):
        rng = np.random.default_rng([SEED, ord(ds.upper()), 7, s])
        h1 = set(rng.permutation(pool)[: len(pool) // 2].tolist())
        # References without H1: a planted gene never meets itself
        refs = {r: [g for g in genes if g not in h1] for r, genes in d["refs"].items()}
        scorer = _make_scorer(universe, refs, mats, scale)
        h1_idx = np.array(sorted(index[g] for g in h1))
        rest = np.setdiff1d(np.arange(len(universe)), h1_idx)
        rng_null = np.random.default_rng([SEED, ord(ds.upper()), 8, s])
        null_picks = np.stack([rest[rng_null.choice(rest.size, 30, replace=False)] for _ in range(args.draws)])
        for k in args.spike_k:
            if k > h1_idx.size:
                continue
            rng_k = np.random.default_rng([SEED, ord(ds.upper()), 9, s, k])
            picks = np.stack([np.concatenate([rest[rng_k.choice(rest.size, 30 - k, replace=False)],
                                              h1_idx[rng_k.choice(h1_idx.size, k, replace=False)]])
                              for _ in range(args.spike_sets)])
            for m in measures:
                _, _, how, m_onts, _ = MEASURES[m]
                null = _simpd(scorer.per_ref(null_picks, how, m_onts))
                obs = _simpd(scorer.per_ref(picks, how, m_onts))
                p = (1 + (null[None, :] >= obs[:, None]).sum(axis=1)) / (1 + null.size)
                hits = int((p < 0.05).sum())
                lo, hi = _wilson(hits, obs.size)
                rows.append({"measure": m, "dataset": ds, "planted": planted, "k": k, "n_sets": obs.size,
                             "frac_p05": hits / obs.size, "ci_low": lo, "ci_high": hi,
                             "median_SimPD": float(np.median(obs)), "null_mean": float(null.mean())})
        print(f"  {ds} spike {planted}: pool {len(pool)}, H1 {len(h1)}", flush=True)
    return rows


def _summary(observed: pd.DataFrame, spike: pd.DataFrame, forms: list) -> pd.DataFrame:
    rows = []
    for m, g in observed.groupby("measure", sort=False):
        row = {"measure": m, "n_sets": len(g)}
        for form in forms:
            f = g[g["form"] == form]
            row[f"{form}_frac_p05"] = float((f["p_expr"] < 0.05).mean()) if len(f) else np.nan
            row[f"{form}_median_p"] = float(f["p_expr"].median()) if len(f) else np.nan
        row["frac_q05"] = float((g["q_expr"] < 0.05).mean())
        if len(spike):
            s = spike[spike["measure"] == m]
            for planted, k, name in (("PD", 0, "typeI_k0"), ("PD", 5, "sens_PD_k5"), ("AD", 5, "AD_k5"),
                                     ("PD", 10, "sens_PD_k10"), ("AD", 10, "AD_k10")):
                v = s[(s["planted"] == planted) & (s["k"] == k)]["frac_p05"]
                row[name] = float(v.mean()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison(spike: pd.DataFrame, reference: str = "M0_simrel_bma") -> pd.DataFrame:
    """Planted genes pooled over the datasets, and the plan's decision rule against the reference measure.

    A measure beats the reference if its type I error (k = 0) is <= 0.075, its
    AD control (k = 5) is not above the reference's, and the lower end of its
    sensitivity interval (PD, k = 5) is above the upper end of the reference's.
    """
    rows = []
    for m, s in spike.groupby("measure", sort=False):
        row = {"measure": m}
        for planted, k in (("PD", 0), ("PD", 2), ("PD", 5), ("PD", 10), ("AD", 5), ("AD", 10)):
            v = s[(s["planted"] == planted) & (s["k"] == k)]
            hits, n = int(round(float((v["frac_p05"] * v["n_sets"]).sum()))), int(v["n_sets"].sum())
            name = "typeI_k0" if k == 0 else f"{planted}_k{k}"
            row[name] = hits / n if n else np.nan
            if name in ("typeI_k0", "PD_k5", "AD_k5"):
                row[f"{name}_low"], row[f"{name}_high"] = _wilson(hits, n)
        rows.append(row)
    out = pd.DataFrame(rows)
    if reference in set(out["measure"]):
        ref = out[out["measure"] == reference].iloc[0]
        out["beats_" + reference] = ((out["typeI_k0"] <= 0.075) & (out["AD_k5"] <= ref["AD_k5"])
                                     & (out["PD_k5_low"] > ref["PD_k5_high"]))
    return out


def _mlp_vs_lora(observed: pd.DataFrame) -> pd.DataFrame:
    """zero-shot+MLP vs LoRA per measure: paired over (dataset, cell type, cut-off), two-sided Wilcoxon."""
    from scipy.stats import wilcoxon
    rows = []
    for m, g in observed.groupby("measure", sort=False):
        w = g.pivot_table(index=["dataset", "cell_type", "selection"], columns="form", values=["SimPD", "p_expr"])
        w = w.dropna(subset=[("SimPD", "mlp"), ("SimPD", "lora")])
        # Positive: LoRA's gene set is more significant (lower p) / more PD-like
        d_logp = np.log10(w[("p_expr", "mlp")]) - np.log10(w[("p_expr", "lora")])
        d_sim = w[("SimPD", "lora")] - w[("SimPD", "mlp")]
        row = {"measure": m, "n_pairs": len(w)}
        for form in ("mlp", "lora"):
            f = g[g["form"] == form]
            row[f"{form}_frac_p05"] = float((f["p_expr"] < 0.05).mean())
            row[f"{form}_median_p"] = float(f["p_expr"].median())
        row["lora_better"] = int((d_logp > 0).sum())
        row["mlp_better"] = int((d_logp < 0).sum())
        row["median_dlog10p"] = float(d_logp.median())
        for ds in sorted(set(w.index.get_level_values("dataset"))):
            row[f"median_dlog10p_{ds}"] = float(d_logp.xs(ds, level="dataset").median())
        row["wilcoxon_p_log10p"] = float(wilcoxon(d_logp, zero_method="pratt").pvalue) if d_logp.any() else 1.0
        row["median_dSimPD"] = float(d_sim.median())
        row["wilcoxon_p_SimPD"] = float(wilcoxon(d_sim, zero_method="pratt").pvalue) if d_sim.any() else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("results/scgpt"))
    p.add_argument("--datasets", nargs="+", default=["A", "B"])
    p.add_argument("--forms", nargs="+", default=FORMS)
    p.add_argument("--measures", nargs="+", default=DEFAULT_MEASURES, choices=list(MEASURES))
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--spike-k", nargs="+", type=int, default=[0, 2, 5, 10])
    p.add_argument("--spike-sets", type=int, default=200, help="Sets per (planted, k); 0 skips the experiment.")
    p.add_argument("--compute-pairs", action="store_true", help="Only fill the pair cache of --measures.")
    p.add_argument("--threads", type=int, default=None, help="go3 threads (default: every core).")
    p.add_argument("--cache-dir", type=Path, default=None, help="Pair cache (default: <results-root>/go3_pair_cache).")
    p.add_argument("--write-pairs-file", type=Path, default=None,
                   help="Only write the genes and reference genes of the run to this JSON.")
    p.add_argument("--pairs-file", type=Path, default=None, help="With --compute-pairs: genes and references from this JSON.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output folder (default: <results-root>/simpd_measures; each run overwrites its CSVs).")
    run(p.parse_args())


if __name__ == "__main__":
    main()
