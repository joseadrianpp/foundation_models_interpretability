#!/usr/bin/env python
"""Check a full run (scripts/run_full.sh) before its results are used.

  * logs: every task started and ended with exit 0, no tracebacks; skipped
    cell types and missing inputs are listed. With several --log-dir (a run
    split over several jobs, oldest first) a task's last end counts;
  * cells: the attention and OLS tasks saw every cell of each cell type;
  * CSSI: Leiden strata per cell type and whether any stratum was valid
    (with none, the CSSI score falls back to the pooled one);
  * gene lists: one per experiment (method x selection x dataset x cell type),
    non-empty, no NaN scores, with the selector fallbacks;
  * post-hoc outputs: enrichment, enrichment summary, GO3 similarity, gene
    matrices, reference gene sets and plots, for every selection.

Prints FAIL / WARN lines and the gene-set sizes, and exits 1 if anything
failed.

  venv/bin/python scripts/check_run.py
  venv/bin/python scripts/check_run.py --datasets A --max-cells-per-type 200
  venv/bin/python scripts/check_run.py --models scgpt --scores ig igsigned --results-dir results/scgpt
  venv/bin/python scripts/check_run.py --forms zero-shot lora                       # a run before the MLP / Step 2.5 forms

--max-cells-per-type and --distances must match the run being checked.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import run_tasks as rt
from pipeline.postprocess.gene_matrix import DEFAULT_DISEASES
from pipeline.postprocess.gene_sets import (CELL_TYPES, FORMS, GENE_LIST_PATHS, METHOD_META, METHODS, MODELS,
                                            SELECTIONS, SUMMARY_PATHS, gene_list_path, methods_for,
                                            output_dir, plot_path)
from pipeline.postprocess.gwas_similarity import DEFAULT_DISTANCES, ONTOLOGIES
from pipeline.postprocess.plots import PD_REFERENCES
from pipeline.steps._step7_common import _slug_cell_type

# Reference gene sets of gwas_similarity (GWAS Catalog traits + the two PD panels)
REFERENCES = ["parkinson", "alzheimer", "cad", "kidney", "panelapp", "parkinsonsuk_ucl"]
# Plots written once per model, and once per model and selection
PLOTS_FIXED = ["gene_set_sizes", "dynamic_selection_cutoffs", "ols_residualization", "selection_comparison"]
PLOTS_PER_SELECTION = (["enrichment_summary", "gwas_set_similarity", "gene_set_overlap"]
                       + [f"gwas_similarity_{r}" for r in PD_REFERENCES])
# Log lines worth a look even when the task exits 0
LOG_WARN_PATTERNS = re.compile(r"skipped:|MISSING|WARNING: GPU")
LOG_FAIL_PATTERNS = re.compile(r"Traceback|CUDA out of memory|Killed|RuntimeError")


class Report:
    def __init__(self):
        self.fails, self.warns = [], []

    def fail(self, msg: str) -> None:
        self.fails.append(msg)
        print(f"  FAIL {msg}")

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"  WARN {msg}")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# Pooled A+B only has the Geneformer zero-shot experiments (see run_tasks.DATASET_DIRS)
DATASET_SCOPE = {"A+B": {("geneformer", "zero-shot")}}
# Models and model forms the checked run covers (--models, --forms)
MODEL_FORMS = [mf for mf, _ in FORMS.values()]
SCORES = ["attention", "ig", "igsigned"]
SCOPE = {"models": list(MODELS), "forms": list(MODEL_FORMS), "scores": list(SCORES), "post": "all"}


def _score_of(name: str) -> str:
    """Score of a task, results folder or method name (attention, ig or igsigned)."""
    return name.split("_", 1)[0]


def _in_scope(dataset: str, model: str, model_form: str) -> bool:
    scope = DATASET_SCOPE.get(dataset)
    return ((scope is None or (model, model_form) in scope)
            and model in SCOPE["models"] and model_form in SCOPE["forms"])


def _methods(dataset: str) -> list:
    return [m for m in METHODS if _in_scope(dataset, METHOD_META[m]["model"], METHOD_META[m]["model_form"])
            and METHOD_META[m]["score"] in SCOPE["scores"]]


def _task_scope(task: str) -> tuple:
    """(model, model form) of a model or OLS task."""
    model = task.rsplit("_", 1)[-1]
    form = next((f for f in rt.FORMS if f"_{f}_" in task), "zero_shot")
    return model, FORMS[form][0]


def _tasks(dataset: str) -> list:
    """Model, OLS and post-hoc tasks a dataset's run should have."""
    if dataset in DATASET_SCOPE:
        return ["attention_geneformer", "attention_residualized_ols_geneformer", "postprocess"]
    post = ["postprocess"] if SCOPE["post"] == "all" else ["gwas_similarity"]
    return [t for t in list(rt.MODEL_TASKS) + list(rt.DERIVED_TASKS)
            if _in_scope(dataset, *_task_scope(t)) and _score_of(t) in SCOPE["scores"]] + post


def check_logs(log_dirs: list, datasets: list, rep: Report) -> None:
    _section(f"logs ({', '.join(str(d) for d in log_dirs)})")
    started, ended, ended_in = set(), {}, {}
    for log_dir in log_dirs:
        main_log = log_dir / "run_full.log"
        if not main_log.exists():
            continue
        text = main_log.read_text()
        started |= set(re.findall(r" start (\S+) (\S+)", text))
        for d, t, code in re.findall(r" end (\S+) (\S+) \(exit (\d+)\)", text):
            ended[(d, t)], ended_in[(d, t)] = int(code), log_dir
    if not started:
        rep.fail("no run_full.log in " + ", ".join(str(d) for d in log_dirs))
        return
    for d in datasets:
        for t in _tasks(d):
            if (d, t) not in started and t in rt.DERIVED_TASKS:
                # Possibly run by hand; its outputs are checked below
                rep.warn(f"{d} {t}: not in run_full.log")
            elif (d, t) not in started:
                rep.fail(f"{d} {t}: never started")
            elif (d, t) not in ended:
                rep.fail(f"{d} {t}: started but did not end (still running or killed)")
            elif ended[(d, t)] != 0:
                rep.fail(f"{d} {t}: exit {ended[(d, t)]}, see {ended_in[(d, t)] / f'{d}_{t}.log'}")

            # The log of the run whose end counts
            if (d, t) not in ended_in or not (ended_in[(d, t)] / f"{d}_{t}.log").exists():
                continue
            task_log = ended_in[(d, t)] / f"{d}_{t}.log"
            for line in task_log.read_text(errors="replace").splitlines():
                if LOG_FAIL_PATTERNS.search(line):
                    rep.fail(f"{task_log.name}: {line.strip()}")
                elif LOG_WARN_PATTERNS.search(line):
                    rep.warn(f"{task_log.name}: {line.strip()}")
    last = log_dirs[-1] / "run_full.log"
    if not (last.exists() and "all done" in last.read_text(errors="replace")):
        rep.warn(f"{last} has no 'all done' line")


def _score_folders() -> list:
    """(model, model form, model task folder with the pooled/CSSI outputs, its OLS folder) of every score."""
    rows = []
    for m in MODELS:
        for form in rt.FORMS:
            model_form = FORMS[form][0]
            rows.append((m, model_form, f"attention_{form}_{m}",
                         f"attention_residualized_ols_{m}" if form == "zero_shot"
                         else f"attention_residualized_ols_{form}_{m}"))
            if form in rt.IG_FORMS.get(m, rt.CLASSIFIER_FORMS):
                rows += [(m, model_form, f"{score}_{form}_{m}", f"{score}_residualized_ols_{form}_{m}")
                         for score in rt.IG_SCORES]
    return rows


SCORE_FOLDERS = _score_folders()


def check_cells_and_cssi(results_dir: Path, dataset: str, max_cells, rep: Report) -> None:
    _section(f"{dataset}: cells, CSSI strata and OLS")
    data_dir = rt.DATASET_DIRS[dataset]
    rows, expected = [], {}
    for model, form, folder, ols_folder in SCORE_FOLDERS:
        if not _in_scope(dataset, model, form) or _score_of(folder) not in SCOPE["scores"]:
            continue
        if model not in expected:
            counts = rt._cell_type_counts(f"attention_{model}", data_dir, max_cells)
            expected[model] = {_slug_cell_type(ct): n for ct, n in counts.items()}
        for ct in CELL_TYPES:
            meta = _read_json(results_dir / folder / ct / "cssi_metadata.json")
            ols = _read_json(results_dir / ols_folder / ct / "ols_summary.json")
            if meta is None:
                rep.fail(f"{folder} {ct}: cssi_metadata.json missing")
                continue
            if ((results_dir / folder / ct / "ig_checkpoint.npz").exists()
                    and not (results_dir / folder / ct / "done.json").exists()):
                rep.fail(f"{folder} {ct}: IG checkpoint left, the task did not finish this cell type")
            if meta["n_cells"] != expected[model].get(ct):
                rep.fail(f"{folder} {ct}: used {meta['n_cells']} cells, the data has {expected[model].get(ct)}")
            if meta["valid_strata"] == 0:
                rep.warn(f"{folder} {ct}: no valid Leiden stratum (>= {rt.CSSI_MIN_CELLS_PER_CONDITION} cells "
                         "per condition), the CSSI score is the pooled one")
            if ols is None:
                rep.fail(f"{ols_folder} {ct}: ols_summary.json missing")
            elif ols["n_cells"] != meta["n_cells"]:
                rep.fail(f"{ols_folder} {ct}: OLS used {ols['n_cells']} cells, the scores {meta['n_cells']}")
            rows.append({"scores": folder, "cell_type": ct, "n_cells": meta["n_cells"],
                         "n_scored": meta.get("n_cells_scored", meta["n_cells"]),
                         "leiden_strata": meta["n_clusters"], "valid_strata": meta["valid_strata"],
                         "ols_r2": None if ols is None else round(ols["r_squared"], 3)})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def gene_list_inventory(results_dir: Path, dataset: str, rep: Report) -> pd.DataFrame:
    _section(f"{dataset}: gene lists")
    rows = []
    for method in _methods(dataset):
        summaries = {ct: _read_json(results_dir / SUMMARY_PATHS[method].format(ct=ct))
                     for ct in CELL_TYPES} if method in SUMMARY_PATHS else {}
        for selection in GENE_LIST_PATHS[method]:
            for ct in CELL_TYPES:
                path = gene_list_path(results_dir, method, selection, ct)
                row = {"dataset": dataset, "method": method, **METHOD_META[method],
                       "selection": selection, "cell_type": ct, "n_genes": 0, "status": "ok", "note": "",
                       "gene_list": str(path.relative_to(results_dir.parent))}
                if not path.exists():
                    row["status"] = "missing"
                    rep.fail(f"{method} {selection} {ct}: {path} missing")
                else:
                    df = pd.read_csv(path)
                    row["n_genes"] = len(df)
                    if df.empty:
                        row["status"] = "empty"
                        rep.fail(f"{method} {selection} {ct}: empty gene list")
                    elif df["score"].isna().any() or df["gene"].duplicated().any():
                        row["status"] = "bad"
                        rep.fail(f"{method} {selection} {ct}: NaN scores or duplicated genes")
                    elif selection == "top30" and len(df) != rt.TOP_N:
                        rep.warn(f"{method} top30 {ct}: {len(df)} genes")
                # Selector fallbacks: z-score kept only the top gene, elbow found no knee
                sel_meta = (summaries.get(ct) or {}).get(selection, {})
                if sel_meta.get("fallback_top1"):
                    row["note"] = "z-score fallback: top gene only"
                elif sel_meta.get("fallback"):
                    row["note"] = f"elbow fallback: {sel_meta['fallback']}"
                if row["note"]:
                    rep.warn(f"{method} {selection} {ct}: {row['note']}")
                rows.append(row)
    inv = pd.DataFrame(rows)
    sizes = inv.pivot_table(index="method", columns=["selection", "cell_type"], values="n_genes",
                            aggfunc="first").reindex(index=_methods(dataset), columns=SELECTIONS, level=0)
    sizes.columns = [f"{s[:2]}:{ct[:5]}" for s, ct in sizes.columns]
    print(sizes.to_string(na_rep="-", float_format=lambda v: f"{v:.0f}"))
    return inv


def check_postprocess(results_dir: Path, dataset: str, inv: pd.DataFrame, distances: list, rep: Report) -> None:
    _section(f"{dataset}: post-hoc outputs")
    refs_dir = results_dir / "gwas_similarity" / "refs"
    for ref in REFERENCES:
        if not (refs_dir / f"{ref}_genes.csv").exists():
            rep.fail(f"reference gene set {ref} missing in {refs_dir}")

    for selection in SELECTIONS:
        present = inv[(inv["selection"] == selection) & (inv["status"] == "ok")]
        n_sets = len(present)

        # Enrichment: one significant-terms CSV per gene set, counted in manifest.csv
        enr_dir = output_dir(results_dir, "enrichment", selection)
        manifest = enr_dir / "manifest.csv"
        if SCOPE["post"] != "all":
            pass
        elif not manifest.exists():
            rep.fail(f"{selection}: {manifest} missing")
        else:
            m = pd.read_csv(manifest)
            if len(m) != n_sets:
                rep.fail(f"{selection}: enrichment manifest has {len(m)} gene sets, expected {n_sets}")
            missing = [f"{r.method}/{r.cell_type}" for r in present.itertuples()
                       if not (enr_dir / r.method / f"{r.cell_type}.csv").exists()]
            if missing:
                rep.fail(f"{selection}: enrichment CSV missing for {', '.join(missing)}")
            no_terms = int((m["n_significant_terms"] == 0).sum())
            print(f"  {selection}: enrichment of {len(m)} gene sets, {no_terms} without significant terms")

        for name in ["summary_table.csv", "pairwise_tests.csv"] if SCOPE["post"] == "all" else []:
            if not (output_dir(results_dir, "enrichment_summary", selection) / name).exists():
                rep.fail(f"{selection}: enrichment_summary/{name} missing")

        # GO3 similarity: one row per (reference, ontology, distance, gene set)
        sim_csv = output_dir(results_dir, "gwas_similarity", selection) / "gene_level_similarity_all.csv"
        if not sim_csv.exists():
            rep.fail(f"{selection}: {sim_csv} missing")
        else:
            sim = pd.read_csv(sim_csv)
            expected = len(REFERENCES) * len(ONTOLOGIES) * len(distances) * n_sets
            if len(sim) != expected or set(sim["distance"]) != set(distances):
                rep.fail(f"{selection}: gwas_similarity has {len(sim)} rows and distances "
                         f"{sorted(set(sim['distance']))}, expected {expected} rows and {sorted(distances)}")
            n_nan = int(sim["similarity"].isna().sum())
            if n_nan:
                rep.warn(f"{selection}: {n_nan} NaN GO3 similarities")

        if SCOPE["post"] != "all":
            continue
        # Gene matrices: one per (cell type, PD reference, distance, ontology)
        gm_dir = output_dir(results_dir, "gene_matrix", selection)
        for name in ["summary_statistics.csv", "method_gene_annotation.csv"]:
            if not (gm_dir / name).exists():
                rep.fail(f"{selection}: gene_matrix/{name} missing")
        n_matrices = len(list((gm_dir / "matrices").glob("*/*.csv")))
        expected = present["cell_type"].nunique() * len(DEFAULT_DISEASES) * len(distances) * len(ONTOLOGIES)
        if n_matrices != expected:
            rep.fail(f"{selection}: {n_matrices} gene matrices, expected {expected}")

        for model in MODELS:
            if not any(_in_scope(dataset, model, METHOD_META[m]["model_form"]) for m in methods_for(selection, model)):
                continue
            for name in PLOTS_PER_SELECTION:
                if not plot_path(results_dir, f"{name}_{model}", selection).exists():
                    rep.fail(f"plot {plot_path(results_dir, f'{name}_{model}', selection).name} missing")
    for model in [m for m in MODELS if SCOPE["post"] == "all" and any(_in_scope(dataset, m, f) for f in MODEL_FORMS)]:
        for name in PLOTS_FIXED:
            if not (results_dir / "plots" / f"{name}_{model}.png").exists():
                rep.fail(f"plot {name}_{model}.png missing")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", choices=list(rt.DATASET_DIRS), default=["A", "B"],
                   help="Datasets to check (default: A B; A+B only when it was run).")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--log-dir", type=Path, nargs="+", default=[ROOT / "logs"],
                   help="Log folder(s) of run_full.sh; several (oldest first) when the run was split "
                        "over several jobs.")
    p.add_argument("--max-cells-per-type", type=int, default=None)
    p.add_argument("--distances", nargs="+", choices=DEFAULT_DISTANCES, default=DEFAULT_DISTANCES)
    p.add_argument("--no-logs", action="store_true", help="Skip the log checks (runs not made by run_full.sh).")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                   help="Models the run covers (default: all; scgpt for a --task scgpt run).")
    p.add_argument("--forms", nargs="+", choices=MODEL_FORMS, default=MODEL_FORMS,
                   help="Model forms the run covers (default: all).")
    p.add_argument("--post", choices=["all", "gwas"], default="all",
                   help="Post-hoc outputs to check: every one, or gwas_similarity only (SimPD).")
    p.add_argument("--scores", nargs="+", choices=SCORES, default=SCORES,
                   help="Scores the run covers (default: all; ig igsigned for the scGPT study).")
    args = p.parse_args()
    SCOPE.update(models=args.models, forms=args.forms, scores=args.scores, post=args.post)

    rep = Report()
    if not args.no_logs:
        check_logs(args.log_dir, args.datasets, rep)
    inventories = []
    for dataset in args.datasets:
        results_dir = args.results_dir / f"dataset_{dataset.lower()}"
        if not results_dir.exists():
            rep.fail(f"{results_dir} missing")
            continue
        check_cells_and_cssi(results_dir, dataset, args.max_cells_per_type, rep)
        inv = gene_list_inventory(results_dir, dataset, rep)
        check_postprocess(results_dir, dataset, inv, args.distances, rep)
        inventories.append(inv)

    if inventories:
        inv = pd.concat(inventories, ignore_index=True)
        n_ok = int((inv["status"] == "ok").sum())
        print(f"\nExperiments with a gene list: {n_ok}/{len(inv)}")
    print(f"{len(rep.fails)} FAIL, {len(rep.warns)} WARN")
    sys.exit(1 if rep.fails else 0)


if __name__ == "__main__":
    main()
