"""Build the experiment table: run column modules, save their parts, join every part.

  python -m pipeline.macrotable.build                       # every module
  python -m pipeline.macrotable.build --modules classifier  # recompute one part, rejoin all
  python -m pipeline.macrotable.build --results-dir results/scgpt --models scgpt \
      --modules similarity faithfulness                     # the scGPT study (then summarize_scgpt.py)

Parts go to <results>/macrotable/parts/<module>.csv and the joined table to
<results>/macrotable/macrotable.csv: KEY columns first, then each module's
columns in MODULES order, and the long gene list (`genes`) last. Modules not run this time are taken from their
saved part, restricted to the current experiments.
"""

import argparse
from pathlib import Path
import re

import pandas as pd

from ..postprocess.gwas_similarity import DEFAULT_DISTANCES
from .columns import classifier, context, enrichment, faithfulness, overlap, similarity
from .experiments import DEFAULT_MODEL_FORMS, KEY, MODEL_LABELS, Settings, load_experiments

# Column modules in table order
MODULES = {
    "overlap": overlap,
    "enrichment": enrichment,
    "similarity": similarity,
    "faithfulness": faithfulness,
    "classifier": classifier,
    "context": context,
}


def _read_part(path: Path) -> pd.DataFrame:
    part = pd.read_csv(path, dtype={c: str for c in KEY})
    if not set(KEY) <= set(part.columns):
        # Saved before a key column existed (Score): its rows cannot be matched
        print(f"[macrotable] WARNING: {path.name} lacks key columns {sorted(set(KEY) - set(part.columns))}, "
              "ignored; recompute it", flush=True)
        return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=KEY))
    part = part.set_index(KEY)
    # Counts (nGenes*, *TermCount, nFeaturesRF) stay integers next to NaN
    counts = [c for c in part.columns if re.match(r"n[A-Z]", c) or c.endswith("TermCount") or c.startswith("CSSI")]
    return part.astype({c: "Int64" for c in counts})


def build(settings: Settings, modules=None, datasets=None, model_forms=DEFAULT_MODEL_FORMS,
          models=None) -> pd.DataFrame:
    experiments = load_experiments(settings.results_root, datasets, model_forms, models)
    print(f"[macrotable] {len(experiments)} experiments", flush=True)
    parts_dir = Path(settings.results_root) / "macrotable" / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    for name in modules or list(MODULES):
        print(f"[macrotable] {name}", flush=True)
        part = MODULES[name].compute(experiments, settings)
        part.to_csv(parts_dir / f"{name}.csv")

    table = pd.DataFrame(index=experiments.index)
    for name in MODULES:
        path = parts_dir / f"{name}.csv"
        if not path.exists():
            print(f"[macrotable] no {name} part yet, its columns are left out", flush=True)
            continue
        part = _read_part(path)
        uncovered = len(experiments.index.difference(part.index))
        if uncovered:
            print(f"[macrotable] WARNING: the saved {name} part lacks {uncovered} experiments (empty in the "
                  f"table); recompute it with --modules {name}", flush=True)
        table = table.join(part.reindex(experiments.index))
    # The gene list is long; last, so it does not push the other columns away
    table = table[[c for c in table.columns if c != "genes"] + [c for c in table.columns if c == "genes"]]
    out = parts_dir.parent / "macrotable.csv"
    table.reset_index().to_csv(out, index=False)
    print(f"[macrotable] {table.shape[0]} rows x {table.shape[1] + len(KEY)} columns -> {out}", flush=True)
    return table


def main():
    p = argparse.ArgumentParser(description="Build the experiment table (macrotable).")
    p.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    p.add_argument("--modules", nargs="+", choices=list(MODULES), default=None,
                   help="Modules to (re)compute; the others are read from their saved parts (default: all).")
    p.add_argument("--datasets", nargs="+", default=None, help="Dataset labels, e.g. A B (default: every one).")
    p.add_argument("--zero-shot-only", action="store_true", help="Only the zero-shot experiments.")
    p.add_argument("--forms", nargs="+", choices=list(DEFAULT_MODEL_FORMS), default=list(DEFAULT_MODEL_FORMS),
                   help="Model forms (default: all four).")
    p.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=None,
                   help="Models (default: all).")
    p.add_argument("--distances", nargs="+", choices=DEFAULT_DISTANCES, default=["SimRel"],
                   help="GO3 distances of the similarity columns; the first has no suffix (default: SimRel).")
    p.add_argument("--rf-max-cells", type=int, default=20000, help="Cells per dataset and cell type for the RF.")
    p.add_argument("--rf-trees", type=int, default=300)
    p.add_argument("--rf-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores of the random forest (default: all).")
    args = p.parse_args()

    settings = Settings(results_root=args.results_dir, distances=args.distances, rf_max_cells=args.rf_max_cells,
                        rf_trees=args.rf_trees, rf_folds=args.rf_folds, n_jobs=args.n_jobs)
    forms = ("zero-shot",) if args.zero_shot_only else tuple(args.forms)
    build(settings, args.modules, args.datasets, forms, args.models)


if __name__ == "__main__":
    main()
