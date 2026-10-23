#!/usr/bin/env python
"""Random baseline of SimPD: is a method's gene set closer to PD than chance?

SimPD of a gene set (as pipeline/macrotable/columns/similarity.py): per
reference, the GO3 set-level BMA similarity (SimRel) in BP, CC and MF,
averaged over the three; then PD GWAS + PanelApp + ParkinsonsUK-UCL minus
Alzheimer + kidney + CAD.

Two nulls per gene set of one dataset and cell type, --draws random sets
each (seed 42), drawn from the cell type's universe (every gene the scGPT
rankings scored there):

  uniform   sets of the same size, any gene of the universe
  expr      sets with the same number of genes in each decile of detection
            rate (1 - dropout in the cell type, from the OLS covariates):
            the rankings favour well-detected genes, which tend to be better
            annotated in GO, so this is the stricter null

Every observed gene set (gwas_similarity's gene_level_similarity_all.csv,
the three selections) is compared with them:

  z         (SimPD - null mean) / null sd
  p         (1 + #draws with SimPD >= observed) / (1 + draws), one-sided

and the mean over the four cell types with the null of that mean (draw i of
every cell type averaged). Gene x gene similarities come from the shared GO3
pair cache (results/scgpt/go3_pair_cache), extended to the whole universe.

Outputs in <results-root>/simpd_baseline/:
  <dataset>_methods.csv     every (method, cell type, selection): SimPD and, per null, mean, sd, z, p
  <dataset>_aggregate.csv   every (method, selection): the mean over the cell types, the same columns

  python simpd_baseline.py --results-root results/scgpt --datasets A B
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.postprocess import _go3_common as g3c
from pipeline.postprocess import gene_sets as gs

SEED = 42
ONTOLOGIES = ["BP", "CC", "MF"]
# Reference -> sign in SimPD (PD references minus the control diseases)
SIGN = {"parkinson": 1, "panelapp": 1, "parkinsonsuk_ucl": 1, "alzheimer": -1, "kidney": -1, "cad": -1}
SELECTION_DIRS = {"top30": "", "z-score": "z-score", "elbow": "elbow"}
NULLS = ("uniform", "expr")
N_BINS = 10
# Detection rate per gene and cell type: the OLS covariates (computed on every cell of the cell type)
COVARIATES = "ig_residualized_ols_zero_shot_scgpt/{ct}/residualized_genes.csv"


def _simpd_table(df: pd.DataFrame) -> pd.DataFrame:
    """SimPD of every (method, cell type) of one gwas_similarity CSV."""
    ref = df.groupby(["method", "cell_type", "reference"]).agg(similarity=("similarity", "mean"),
                                                                n_genes=("n_genes", "first")).reset_index()
    ref["signed"] = ref["reference"].map(SIGN) * ref["similarity"]
    return ref.groupby(["method", "cell_type"]).agg(SimPD=("signed", "sum"), n_genes=("n_genes", "first")).reset_index()


def _universe(results_dir: Path, cell_type: str) -> list:
    """Every gene a scGPT IG ranking scored in this cell type."""
    genes = set()
    for method in gs.METHODS:
        meta = gs.METHOD_META[method]
        if meta["model"] != "scgpt" or meta["score"] == "attention" or method not in gs.ALL_SCORE_PATHS:
            continue
        path = results_dir / gs.ALL_SCORE_PATHS[method].format(ct=cell_type)
        if path.exists():
            genes |= set(pd.read_csv(path, usecols=["gene"])["gene"].astype(str))
    return sorted(genes)


class SetScorer:
    """SimPD of gene sets of one universe (indices into it), from the gene x reference similarities."""

    def __init__(self, universe: list, refs: dict, pms: dict):
        self.blocks, self.best = {}, {}
        for o in ONTOLOGIES:
            rows = np.array([pms[o]._gi[g] for g in universe])
            for r, genes in refs.items():
                block = pms[o].values[np.ix_(rows, [pms[o]._ri[g] for g in genes])]
                self.blocks[(o, r)] = block
                # Each universe gene's best match in the reference (one half of the BMA)
                self.best[(o, r)] = block.max(axis=1)

    def simpd(self, pick: np.ndarray) -> float:
        out = 0.0
        for r, sign in SIGN.items():
            # BMA as g3c.bma: best matches of the set's genes and of the reference's genes
            sims = [(self.best[(o, r)][pick].sum() + self.blocks[(o, r)][pick].max(axis=0).sum())
                    / (pick.size + self.blocks[(o, r)].shape[1]) for o in ONTOLOGIES]
            out += sign * float(np.nanmean(sims))
        return out


def _detection_bins(results_dir: Path, cell_type: str, universe: list) -> np.ndarray:
    """Decile of detection rate of every universe gene (genes without covariates: the lowest)."""
    cov = pd.read_csv(results_dir / COVARIATES.format(ct=cell_type), usecols=["gene", "dropout_rate"])
    detection = 1.0 - cov.set_index(cov["gene"].astype(str))["dropout_rate"]
    det = detection.reindex(universe).fillna(0.0).to_numpy()
    edges = np.quantile(det, np.linspace(0, 1, N_BINS + 1)[1:-1])
    return np.searchsorted(edges, det, side="right")


def _summary(obs: float, v: np.ndarray, name: str) -> dict:
    sd = v.std(ddof=1)
    return {f"{name}_mean": v.mean(), f"{name}_sd": sd, f"{name}_z": (obs - v.mean()) / sd,
            f"{name}_p": (1 + (v >= obs).sum()) / (1 + v.size)}


def run(results_root: Path, dataset: str, draws: int) -> None:
    results_dir = results_root / f"dataset_{dataset.lower()}"
    out_dir = results_root / "simpd_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = results_dir / "gwas_similarity" / "refs"
    refs = {**g3c.fetch_all_gwas(refs_dir), **g3c.load_pd_panels(refs_dir)}

    # Observed SimPD of every gene set, per selection
    obs = []
    for selection, sub in SELECTION_DIRS.items():
        df = pd.read_csv(results_dir / "gwas_similarity" / sub / "gene_level_similarity_all.csv")
        obs.append(_simpd_table(df[df["distance"] == "SimRel"]).assign(selection=selection))
    obs = pd.concat(obs, ignore_index=True)

    # Gene x gene similarities over every cell type's universe (only the missing pairs are computed)
    universes = {ct: _universe(results_dir, ct) for ct in gs.CELL_TYPES}
    print(f"{dataset}: universes { {ct: len(u) for ct, u in universes.items()} }", flush=True)
    counter = g3c.init_go3()
    all_genes = sorted(set().union(*universes.values()))
    all_refs = sorted({g for genes in refs.values() for g in genes})
    cache_dir = g3c.pair_cache_dir(results_dir)
    pms = {o: g3c.pair_similarities(all_genes, all_refs, o, "SimRel", counter, cache_dir) for o in ONTOLOGIES}

    # Both nulls of every observed set: draws[(method, selection, cell type)][null] = (draws,) SimPD
    null_draws, rows = {}, []
    for i, ct in enumerate(gs.CELL_TYPES):
        universe = universes[ct]
        index = {g: k for k, g in enumerate(universe)}
        scorer = SetScorer(universe, refs, pms)
        bins = _detection_bins(results_dir, ct, universe)
        members = [np.where(bins == b)[0] for b in range(N_BINS)]
        uniform = {}
        sub = obs[obs["cell_type"] == ct]
        print(f"  {ct}: {len(sub)} gene sets x {draws} draws", flush=True)
        for r in sub.itertuples(index=False):
            genes = gs.load_genes(results_dir, r.method, r.selection, ct)
            pick = np.array([index[g] for g in genes if g in index])
            # Uniform: one null per set size, shared by the sets of that size
            if r.n_genes not in uniform:
                rng = np.random.default_rng([SEED, ord(dataset.upper()), i, 0, r.n_genes])
                uniform[r.n_genes] = np.array([scorer.simpd(rng.choice(len(universe), r.n_genes, replace=False))
                                               for _ in range(draws)])
            # Expression-matched: as many genes per detection decile as the set has
            counts = np.bincount(bins[pick], minlength=N_BINS)
            rng = np.random.default_rng([SEED, ord(dataset.upper()), i, 1, *counts.tolist()])
            expr = np.array([scorer.simpd(np.concatenate([rng.choice(members[b], c, replace=False)
                                                          for b, c in enumerate(counts) if c]))
                             for _ in range(draws)])
            null_draws[(r.method, r.selection, ct)] = {"uniform": uniform[r.n_genes], "expr": expr}
            rows.append({**r._asdict(), **_summary(r.SimPD, uniform[r.n_genes], "uniform"),
                         **_summary(r.SimPD, expr, "expr")})
    pd.DataFrame(rows).to_csv(out_dir / f"{dataset}_methods.csv", index=False)

    # Mean over the cell types against the null of that mean (draw i of every cell type averaged)
    agg = []
    for (method, selection), g in obs.groupby(["method", "selection"]):
        if len(g) != len(gs.CELL_TYPES):
            continue
        m = g["SimPD"].mean()
        row = {"method": method, "selection": selection, "SimPD": m}
        for name in NULLS:
            v = np.mean([null_draws[(method, selection, ct)][name] for ct in g["cell_type"]], axis=0)
            row.update(_summary(m, v, name))
        agg.append(row)
    agg = pd.DataFrame(agg)
    agg.to_csv(out_dir / f"{dataset}_aggregate.csv", index=False)
    print(f"\n# {dataset}: SimPD against random gene sets (mean over cell types)\n"
          f"{agg.round(3).to_string(index=False)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("results/scgpt"))
    p.add_argument("--datasets", nargs="+", default=["A", "B"])
    p.add_argument("--draws", type=int, default=1000)
    args = p.parse_args()
    for dataset in args.datasets:
        run(args.results_root, dataset, args.draws)


if __name__ == "__main__":
    main()
