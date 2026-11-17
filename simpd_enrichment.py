#!/usr/bin/env python
"""SimPD from GO enrichment as JA proposes: compare the enriched GO terms of a gene set with the references'.

JA's termset_similarity task (his main branch,
pipeline/postprocess/termset_similarity.py, 2026-10-23), with his options:
each gene set and each reference is turned into its significant GO terms with
g:Profiler (GO:BP, GO:CC, GO:MF, no electronic annotations, g:SCS), obsolete
terms left out, and the set's terms are compared with each reference's per
ontology with GO3 (SimRel term similarity, best-match average); a
reference's similarity is the mean over the ontologies with terms on both
sides, and SimPD keeps its formula (mean of the PD references minus mean of
the control diseases, simpd_measures.py). Backgrounds: a gene set against the
genes its own pooled |IG| ranking scores (inside scGPT's gene_names.csv of the
dataset), a reference against that gene_names.csv. Plan:
docs/plan_simpd_lora_2026-11.md, section 5; results:
docs/distancias_lora_enriquecimiento.md, section 6.

There is no p-value: only a few random 30-gene sets have any significant term,
so a null of random sets is mostly undefined. The planted-gene sets are judged
without a null instead: the AUC of their SimPD against the k = 0 sets.

Planted genes are drawn exactly as simpd_measures.py does (same seeds; with
fewer sets, the first ones of its sequences): random 30-gene sets of the
dataset's universe with k genes from one half (H1) of the PD (or AD) reference
genes, scored against the references without H1. They have no ranking, so
their background is the dataset's universe.

g:Profiler answers are cached per gene set in
<out-dir>/gprofiler_cache/<mode>__<background>.jsonl (appended per batch, so an
interrupted run resumes), with the g:Profiler version in manifest.json.

Outputs in <results-root>/simpd_enrichment/ (--out-dir):
  observed.csv     per gene set: significant terms per ontology (n_BP, n_CC,
                   n_MF), covered (any term), SimPDGWAS ... SimCAD (one per
                   reference, the macrotable's names) and SimPD
  mlp_vs_lora.csv  zero-shot+MLP vs LoRA in the raw SimPD (paired over dataset
                   x cell type x cut-off, pairs where both sets have terms,
                   two-sided Wilcoxon), coverage per form
  planted_separation.csv  per measure (M0, the current SimPD, and E0),
                   dataset, planted (PD / AD) and k: the fraction of sets with
                   terms (coverage), the median SimPD and the AUC of their
                   SimPD against the k = 0 sets of the same series (0.5 = no
                   separation; PD should be > 0.5, AD not)

  python simpd_enrichment.py --results-root results/scgpt
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import go3
import numpy as np
import pandas as pd
from gprofiler import GProfiler

from pipeline import config as C
from pipeline.postprocess import _go3_common as g3c
from pipeline.postprocess import gene_sets as gs
from pipeline.utils.progress import progress
from simpd_baseline import SIGN, SEED, _universe
from simpd_measures import (FORM_LABELS, FORMS, REF_COLUMNS, ONTOLOGIES as M_ONTOLOGIES, PD_REFS, REF_ORDER, Scorer,
                            _gene_sets, _load_mats, _refs, _simpd)

SOURCES = {"GO:BP": "BP", "GO:CC": "CC", "GO:MF": "MF"}
ONTOLOGIES = list(SOURCES.values())
MEASURE = "E0"
PROFILE = dict(organism="hsapiens", sources=list(SOURCES), no_iea=True, significance_threshold_method="g_SCS")


# ---------------------------------------------------------------------------
# g:Profiler, cached per gene set
# ---------------------------------------------------------------------------

def _key(genes) -> str:
    return hashlib.sha1("|".join(sorted(set(genes))).encode()).hexdigest()


def _parse(df: pd.DataFrame) -> dict:
    """{"sig": {ont: significant terms}} of one query's rows."""
    out = {"sig": {o: [] for o in ONTOLOGIES}}
    if df is None or not len(df):
        return out
    for source, o in SOURCES.items():
        sub = df[df["source"] == source]
        out["sig"][o] = sorted(sub.loc[sub["significant"].astype(bool), "native"].tolist())
    return out


class Enricher:
    """g:Profiler answers per gene set, background and mode, cached on disk.

    mode "sig": the significant terms only; mode "ranked" (every term with an
    overlap, all_results) is only read from caches of earlier runs, whose
    significant terms are the same.
    """

    def __init__(self, cache_dir: Path, batch: int, workers: int):
        self.cache_dir, self.batch, self.workers = cache_dir, batch, workers
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache, self.version = {}, None

    def _file(self, mode: str, bg_key: str) -> Path:
        return self.cache_dir / f"{mode}__{bg_key}.jsonl"

    def _load(self, mode: str, bg_key: str) -> dict:
        if (mode, bg_key) not in self.cache:
            path, entries = self._file(mode, bg_key), {}
            if path.exists():
                for line in open(path):
                    e = json.loads(line)
                    entries[e["key"]] = e
            self.cache[(mode, bg_key)] = entries
        return self.cache[(mode, bg_key)]

    def _query(self, sets: list, background, mode: str) -> list:
        gp = GProfiler(return_dataframe=True)
        query = {f"q{i}": sorted(set(g)) for i, g in enumerate(sets)}
        scope = dict(domain_scope="custom", background=background) if background else dict(domain_scope="annotated")
        for attempt in range(6):
            try:
                df = gp.profile(query=query, all_results=(mode == "ranked"), **scope, **PROFILE)
                break
            except Exception as err:  # network / rate limit: back off and retry
                if attempt == 5:
                    raise
                print(f"  g:Profiler error ({err}); retry in {30 * (attempt + 1)} s", flush=True)
                time.sleep(30 * (attempt + 1))
        self.version = gp.meta.get("version", self.version)
        return [_parse(df[df["query"] == name] if len(df) else df) for name in query]

    def fetch(self, sets: list, background, mode: str, desc: str) -> None:
        """Query every set not cached yet (batches of self.batch, self.workers in parallel)."""
        bg_key = _key(background)[:16] if background else "genome"
        entries = self._load(mode, bg_key)
        todo = {}
        for genes in sets:
            k = _key(genes)
            if k not in entries:
                todo.setdefault(k, sorted(set(genes)))
        if not todo:
            return
        keys = list(todo)
        batches = [keys[i:i + self.batch] for i in range(0, len(keys), self.batch)]
        bar = progress(total=len(keys), desc=f"g:Profiler {desc}", unit="set")
        path = self._file(mode, bg_key)

        def run(batch_keys):
            return batch_keys, self._query([todo[k] for k in batch_keys], background, mode)

        with ThreadPoolExecutor(self.workers) as pool, open(path, "a") as handle:
            for batch_keys, results in pool.map(run, batches):
                for k, res in zip(batch_keys, results):
                    e = {"key": k, **res}
                    entries[k] = e
                    handle.write(json.dumps(e) + "\n")
                handle.flush()
                bar.update(len(batch_keys))
        bar.close()

    def get(self, genes, background) -> dict:
        """Significant terms of a set: from the "sig" cache, or from an earlier "ranked" answer."""
        bg_key = _key(background)[:16] if background else "genome"
        k = _key(genes)
        sig = self._load("sig", bg_key)
        return sig[k] if k in sig else self._load("ranked", bg_key)[k]

    def fetch_missing(self, sets: list, background, desc: str) -> None:
        """Query in "sig" mode only the sets neither cache has."""
        bg_key = _key(background)[:16] if background else "genome"
        ranked = self._load("ranked", bg_key)
        self.fetch([g for g in sets if _key(g) not in ranked], background, "sig", desc)


# ---------------------------------------------------------------------------
# Term-set similarity (SimRel, BMA) from one term x term matrix per ontology
# ---------------------------------------------------------------------------

class TermScorer:
    def __init__(self, counter, set_terms: dict, ref_terms: dict, chunk: int = 2_000_000):
        self.rows, self.cols, self.mats = {}, {}, {}
        for o in ONTOLOGIES:
            rows, cols = sorted(set_terms[o]), sorted(ref_terms[o])
            self.rows[o] = {t: i for i, t in enumerate(rows)}
            self.cols[o] = {t: j for j, t in enumerate(cols)}
            mat = np.zeros((len(rows), len(cols)), np.float32)
            step = max(1, chunk // max(1, len(cols)))
            start = time.time()
            for s in progress(range(0, len(rows), step), desc=f"go3 {o} terms", unit="block"):
                block = rows[s:s + step]
                a = [t for t in block for _ in cols]
                b = cols * len(block)
                mat[s:s + len(block)] = np.asarray(go3.batch_similarity(a, b, "SimRel", counter),
                                                   dtype=np.float32).reshape(len(block), len(cols))
            self.mats[o] = mat
            print(f"  {o}: {len(rows)} set terms x {len(cols)} reference terms in {time.time() - start:.0f} s",
                  flush=True)

    def sim(self, terms: dict, ref: dict) -> float:
        """Mean over the ontologies with terms on both sides of the BMA of the term x term block."""
        vals = []
        for o in ONTOLOGIES:
            if not terms[o] or not ref[o]:
                continue
            block = self.mats[o][np.ix_([self.rows[o][t] for t in terms[o]], [self.cols[o][t] for t in ref[o]])]
            vals.append((block.max(axis=1).sum() + block.max(axis=0).sum()) / (block.shape[0] + block.shape[1]))
        return float(np.mean(vals)) if vals else np.nan

    def simpd(self, terms: dict, refs: dict) -> tuple:
        per_ref = {r: self.sim(terms, refs[r]) for r in REF_ORDER}
        return sum(SIGN[r] * v for r, v in per_ref.items()), per_ref


# ---------------------------------------------------------------------------
# JA's backgrounds (pipeline/postprocess/termset_similarity.py of his main)
# ---------------------------------------------------------------------------

def _model_universe(results_root: Path, ds: str) -> list:
    """Every gene of scGPT's gene_names.csv in the dataset (JA's _universe)."""
    path = results_root.resolve().parents[1] / "data" / f"dataset_{ds.lower()}" / C.SCGPT_GENE_NAMES
    return sorted(set(pd.read_csv(path)["gene"].astype(str)))


def _ranking_background(results_dir: Path, form: str, cell_type: str, universe: set) -> list:
    """The genes the form's pooled |IG| ranking scores in the cell type, inside the model's universe."""
    path = results_dir / gs.ALL_SCORE_PATHS[f"ig_{form}_scgpt"].format(ct=cell_type)
    return sorted(set(pd.read_csv(path, usecols=["gene"])["gene"].astype(str)) & universe)


def _auc(vals: np.ndarray, base: np.ndarray) -> float:
    """P(planted set > k = 0 set), ties count one half (Mann-Whitney U / (n1 n2)), over the sets with a value."""
    from scipy.stats import mannwhitneyu
    ok, base = vals[~np.isnan(vals)], base[~np.isnan(base)]
    return float(mannwhitneyu(ok, base).statistic / (ok.size * base.size)) if ok.size and base.size else np.nan


def _planted_separation(values: dict) -> pd.DataFrame:
    rows = []
    for (ds, planted), v in values.items():
        for k in sorted(v):
            for m, vals in v[k].items():
                ok = vals[~np.isnan(vals)]
                rows.append({"measure": m, "dataset": ds, "planted": planted, "k": k, "n_sets": vals.size,
                             "coverage": ok.size / vals.size,
                             "median_SimPD": float(np.median(ok)) if ok.size else np.nan,
                             "auc_vs_k0": _auc(vals, v[0][m])})
    return pd.DataFrame(rows)


def _mlp_vs_lora(observed: pd.DataFrame) -> pd.DataFrame:
    """zero-shot+MLP vs LoRA in the raw SimPD (no null), pairs where both sets have terms."""
    from scipy.stats import wilcoxon
    w = observed.pivot_table(index=["dataset", "cell_type", "selection"], columns="form", values="SimPD")
    d = (w["lora"] - w["mlp"]).dropna()
    row = {"measure": MEASURE, "n_pairs": len(d), "lora_higher": int((d > 0).sum()), "mlp_higher": int((d < 0).sum()),
           "median_diff": float(d.median()),
           "wilcoxon_p": float(wilcoxon(d, zero_method="pratt").pvalue) if d.any() else 1.0}
    for ds in sorted(set(d.index.get_level_values("dataset"))):
        dd = d.xs(ds, level="dataset")
        row[f"lora_higher_{ds}"], row[f"mlp_higher_{ds}"] = int((dd > 0).sum()), int((dd < 0).sum())
    for f in ("mlp", "lora"):
        row[f"coverage_{f}"] = float(observed[observed["form"] == f]["covered"].mean())
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    out_dir = args.out_dir or args.results_root / "simpd_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    enr = Enricher(out_dir / "gprofiler_cache", args.batch, args.workers)
    counter = g3c.init_go3(args.threads)

    # JA's _in_go3: terms GO3's ontology knows (g:Profiler's GO release may differ) and not obsolete
    known = {}

    def ok(t):
        if t not in known:
            try:
                known[t] = not go3.get_term_by_id(t).is_obsolete
            except ValueError:
                known[t] = False
        return known[t]

    def terms_of(entry):
        return {o: [t for t in entry["sig"][o] if ok(t)] for o in ONTOLOGIES}

    # 1. The gene sets: observed sets (inside scGPT's genes) and planted sets
    jobs = []  # (dataset, cell type, form, selection, genes in the cell type's universe, those inside scGPT's genes)
    spikes = []  # (dataset, planted, k, sets, union universe, references without H1)
    datasets = {}
    for ds in args.datasets:
        rd = args.results_root / f"dataset_{ds.lower()}"
        refs = _refs(rd)
        universes = {ct: _universe(rd, ct) for ct in gs.CELL_TYPES}
        model_universe = _model_universe(args.results_root, ds)
        inside = set(model_universe)
        datasets[ds] = {"model_universe": model_universe,
                        "ranking_bg": {(f, ct): _ranking_background(rd, f, ct, inside)
                                       for f in args.forms for ct in gs.CELL_TYPES},
                        "refs_inside": {r: [g for g in genes if g in inside] for r, genes in refs.items()}}
        for form, selection, ct, genes in _gene_sets(rd, args.forms):
            universe = set(universes[ct])
            genes = [g for g in genes if g in universe]
            jobs.append((ds, ct, form, selection, genes, [g for g in genes if g in inside]))
        if args.spike_sets:
            union = sorted(set().union(*universes.values()))
            pools = {"PD": sorted({g for r in PD_REFS for g in refs[r]} & set(union)),
                     "AD": sorted(set(refs["alzheimer"]) & set(union))}
            for s, (planted, pool) in enumerate(pools.items()):
                # Same draws as simpd_measures._spike
                rng = np.random.default_rng([SEED, ord(ds.upper()), 7, s])
                h1 = set(rng.permutation(pool)[: len(pool) // 2].tolist())
                refs_wo = {r: [g for g in genes if g not in h1] for r, genes in refs.items()}
                h1_idx = np.array(sorted(union.index(g) for g in h1))
                rest = np.setdiff1d(np.arange(len(union)), h1_idx)
                for k in args.spike_k:
                    rng_k = np.random.default_rng([SEED, ord(ds.upper()), 9, s, k])
                    picked = [[union[j] for j in np.concatenate([rest[rng_k.choice(rest.size, 30 - k, replace=False)],
                                                                 h1_idx[rng_k.choice(h1_idx.size, k, replace=False)]])]
                              for _ in range(args.spike_sets)]
                    spikes.append((ds, planted, k, picked, union, refs_wo))

    # 2. g:Profiler: references against scGPT's genes, each set against its ranking's genes,
    #    planted sets against the dataset's universe
    for ds, d in datasets.items():
        inside = set(d["model_universe"])
        enr.fetch(list(d["refs_inside"].values()), d["model_universe"], "sig", f"{ds} references")
        for planted in ("PD", "AD"):
            mine = [s for s in spikes if s[0] == ds and s[1] == planted]
            if mine:
                refs_wo = mine[0][5]
                enr.fetch([[g for g in genes if g in inside] for genes in refs_wo.values()], d["model_universe"],
                          "sig", f"{ds} references without H1 {planted}")
                enr.fetch_missing([g for s in mine for g in s[3]], mine[0][4], f"{ds} planted {planted}")
        for (form, ct), bg in d["ranking_bg"].items():
            enr.fetch([j[5] for j in jobs if j[0] == ds and j[1] == ct and j[2] == form], bg, "sig",
                      f"{ds} {form} {ct}")

    # 3. One term x term SimRel matrix per ontology
    ref_terms = {}
    for ds, d in datasets.items():
        inside = set(d["model_universe"])
        for r, genes in d["refs_inside"].items():
            ref_terms[(ds, r, "full")] = terms_of(enr.get(genes, d["model_universe"]))
        for planted in ("PD", "AD"):
            mine = [s for s in spikes if s[0] == ds and s[1] == planted]
            for r, genes in (mine[0][5].items() if mine else []):
                ref_terms[(ds, r, planted)] = terms_of(enr.get([g for g in genes if g in inside], d["model_universe"]))
    set_terms = {}
    for ds, ct, form, selection, _, genes in jobs:
        set_terms[(ds, ct, form, selection)] = terms_of(enr.get(genes, datasets[ds]["ranking_bg"][(form, ct)]))
    planted_terms = {(ds, planted, k): [terms_of(enr.get(g, union)) for g in picked]
                     for ds, planted, k, picked, union, _ in spikes}
    set_vocab = {o: set() for o in ONTOLOGIES}
    ref_vocab = {o: set() for o in ONTOLOGIES}
    for t in ref_terms.values():
        for o in ONTOLOGIES:
            ref_vocab[o] |= set(t[o])
    for t in list(set_terms.values()) + [t for ts in planted_terms.values() for t in ts]:
        for o in ONTOLOGIES:
            set_vocab[o] |= set(t[o])
    print("\n# term x term SimRel", flush=True)
    scorer = TermScorer(counter, set_vocab, ref_vocab)

    # 4. Observed sets
    rows = []
    for ds, ct, form, selection, genes, _ in jobs:
        t = set_terms[(ds, ct, form, selection)]
        value, per_ref = scorer.simpd(t, {r: ref_terms[(ds, r, "full")] for r in REF_ORDER})
        rows.append({"measure": MEASURE, "dataset": ds, "form": form, "cell_type": ct, "selection": selection,
                     "n_genes": len(genes), **{f"n_{o}": len(t[o]) for o in ONTOLOGIES},
                     "covered": bool(any(t[o] for o in ONTOLOGIES)),
                     **{REF_COLUMNS[r]: v for r, v in per_ref.items()}, "SimPD": value})
    observed = pd.DataFrame(rows)
    observed["form"] = observed["form"].replace(FORM_LABELS)
    observed.to_csv(out_dir / "observed.csv", index=False)
    cover = observed.groupby(["form", "selection"])["covered"].mean().unstack()
    print(f"\n# Coverage (sets with terms)\n{cover.round(2)}", flush=True)
    if {"mlp", "lora"} <= set(args.forms):
        pairs = _mlp_vs_lora(observed)
        pairs.to_csv(out_dir / "mlp_vs_lora.csv", index=False)
        print(f"\n# zero-shot+MLP vs LoRA, raw SimPD (no null)\n{pairs.round(3).T.to_string()}", flush=True)

    # 5. Planted genes: M0 and E0, AUC against k = 0
    if spikes:
        pairs_file = json.load(open(args.results_root / "simpd_measures" / "pairs.json"))
        mats = _load_mats("SimRel", "bma", M_ONTOLOGIES, pairs_file["genes"], pairs_file["refs"], counter,
                          args.results_root / "go3_pair_cache")
        values = {}
        for ds, planted, k, picked, union, refs_wo in spikes:
            index = {g: i for i, g in enumerate(union)}
            m0 = Scorer(union, refs_wo, mats, {o: 1.0 for o in M_ONTOLOGIES})
            picks = np.array([[index[g] for g in genes] for genes in picked])
            refs = {r: ref_terms[(ds, r, planted)] for r in REF_ORDER}
            values.setdefault((ds, planted), {})[k] = {
                "M0_simrel_bma": _simpd(m0.per_ref(picks, "bma", M_ONTOLOGIES)),
                MEASURE: np.array([scorer.simpd(t, refs)[0] for t in planted_terms[(ds, planted, k)]])}
        sep = _planted_separation(values)
        sep.to_csv(out_dir / "planted_separation.csv", index=False)
        pooled = sep[sep["k"] > 0].groupby(["measure", "planted", "k"], sort=False)[["coverage", "auc_vs_k0"]].mean()
        print(f"\n# Planted sets without a null: AUC against k = 0 (mean of A and B)\n"
              f"{pooled.unstack(['planted', 'k']).round(3).to_string()}", flush=True)

    # g:Profiler's version: from this run's queries, or kept from the run that filled the cache
    manifest = out_dir / "manifest.json"
    version = enr.version or (json.load(open(manifest)).get("gprofiler_version") if manifest.exists() else None)
    json.dump({"measure": MEASURE, "datasets": args.datasets, "forms": args.forms, "spike_k": args.spike_k,
               "spike_sets": args.spike_sets, "seed": SEED, "gprofiler_version": version,
               "gprofiler_options": PROFILE,
               "go_terms_left_out": sorted(t for t, v in known.items() if not v)},
              open(manifest, "w"), indent=1)
    print(f"\noutputs in {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("results/scgpt"))
    p.add_argument("--datasets", nargs="+", default=["A", "B"])
    p.add_argument("--forms", nargs="+", default=FORMS)
    p.add_argument("--spike-k", nargs="+", type=int, default=[0, 2, 5, 10])
    p.add_argument("--spike-sets", type=int, default=100, help="Planted sets per (planted, k); 0 skips the experiment.")
    p.add_argument("--batch", type=int, default=50, help="Gene sets per g:Profiler request.")
    p.add_argument("--workers", type=int, default=2, help="g:Profiler requests in parallel.")
    p.add_argument("--threads", type=int, default=None, help="go3 threads (default: every core).")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <results-root>/simpd_enrichment.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
