#!/usr/bin/env python
"""Result tables of the scGPT study: FGCell and SimPD per form, score, ranking and cut-off.

Reads <results-root>/<macrotable>/macrotable.csv (--macrotable, default
macrotable), built with
`python -m pipeline.macrotable.build --results-dir <results-root> --models scgpt
--modules similarity faithfulness`, and writes to <results-root>/<tables-dir>/
(default tables; only the --scores rows, default IG):

  <dataset>_<cell type>.csv   one row per (form, score, ranking, TopN): SimPD
                              of the gene set (Problem C) and FGCell, the
                              faithfulness of the form's per-cell IG (Problem A)
  <dataset>_aggregate.csv     the same rows, mean over the four cell types
  <dataset>_fgcell.csv        FGCell of every form's per-cell explanation
                              (rows) per cell type, and their mean (columns);
                              the per-cell IG of the cases / controls rankings
                              on the cells of that condition only
  fgcell_vs_simpd.csv         one row per (form, score, ranking): FGCell and
                              the SimPD of each cut-off (its p against chance
                              with --with-baseline), both datasets

With --with-baseline and simpd_baseline.py's outputs
(<results-root>/simpd_baseline/), the SimPD rows also get its comparison with
random gene sets: SimPD_p, the one-sided empirical p against sets drawn with
the same detection rates, SimPD_q, its Benjamini-Hochberg q over the rows of
the dataset's table, and SimPD_p_uniform, against sets of any genes; the
aggregate rows, against the null of the mean over the cell types. p-values,
not z (user's choice, 2026-10-31). Without it (the default) the tables report
SimPD and FGCell only. The FG of the population rankings is not reported
since 2026-10-31 (near chance for the classifiers). The self-supervised form
is supplementary: its FGCell explains an embedding magnitude, not a
probability, and can exceed 1, so it is not comparable with the classifiers'.

The attention rows are left out (the study dropped raw attention) unless
--with-attention.

  python summarize_scgpt.py --results-root results/scgpt
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROWS = ["ModelForm", "Score", "Ranking", "TopN"]
VALUES = ["SimPD", "FGCell"]
BASELINE = ["SimPD_p", "SimPD_q", "SimPD_p_uniform"]
# simpd_baseline.py's names -> the macrotable's labels
FORM_LABELS = {"zero_shot": "zero-shot", "step2_5": "self-supervised", "mlp": "zero-shot+MLP", "lora": "LoRA"}
SCORE_LABELS = {"ig": "IG", "igsigned": "IG±", "rollout": "rollout", "rolloutsigned": "rollout±",
                "gradrollout": "gradrollout", "gradrolloutsigned": "gradrollout±"}
RANKING_LABELS = {None: "pooled", "cssi_": "CSSI", "residualized_ols_": "residualized", "cases_": "cases",
                  "controls_": "controls"}
CELL_TYPE_LABELS = {"astrocytes": "astrocyte", "da_neurons": "neuron", "microglia": "microglia",
                    "oligodendrocytes": "oligodendrocyte"}
TOPN_LABELS = {"top30": "top30", "z-score": "zscore", "elbow": "elbow"}
# Table order of the key values
ORDER = {"ModelForm": ["zero-shot+MLP", "LoRA", "self-supervised", "zero-shot"],
         "Score": ["IG", "IG±", "rollout", "rollout±", "gradrollout", "gradrollout±", "attention"],
         "Ranking": ["pooled", "cases", "controls", "CSSI", "residualized"],
         "TopN": ["top30", "zscore", "elbow"]}


def _sorted(df: pd.DataFrame, cols) -> pd.DataFrame:
    keys = {c: df[c].map({v: i for i, v in enumerate(ORDER[c])}) for c in cols}
    return df.assign(**{f"_{c}": k for c, k in keys.items()}).sort_values([f"_{c}" for c in cols]) \
        .drop(columns=[f"_{c}" for c in cols])


def _bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q of the non-missing p-values (NaN stays NaN)."""
    valid = p.dropna()
    n = len(valid)
    if n == 0:
        return p.copy()
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order] * n / np.arange(1, n + 1)
    # Monotone from the largest p down, capped at 1
    q_sorted = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    q = np.empty(n)
    q[order] = q_sorted
    return pd.Series(q, index=valid.index).reindex(p.index)


def _baseline(results_root: Path, dataset: str, part: str):
    """simpd_baseline.py's comparison with chance (part: methods or aggregate), in macrotable labels."""
    path = results_root / "simpd_baseline" / f"{dataset}_{part}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keys = df["method"].str.extract(r"^(igsigned|ig|gradrolloutsigned|gradrollout|rolloutsigned|rollout)_(cssi_|residualized_ols_|cases_|controls_)?(.*)_scgpt$")
    out = pd.DataFrame({"ModelForm": keys[2].map(FORM_LABELS), "Score": keys[0].map(SCORE_LABELS),
                        "Ranking": keys[1].map(lambda r: RANKING_LABELS[r if isinstance(r, str) else None]),
                        "TopN": df["selection"].map(TOPN_LABELS),
                        "SimPD_p": df["expr_p"], "SimPD_p_uniform": df["uniform_p"]})
    if "cell_type" in df:
        out.insert(0, "CellType", df["cell_type"].map(CELL_TYPE_LABELS))
    return out


def _fgcell_vs_simpd(aggregates: dict) -> pd.DataFrame:
    """One row per (form, score, ranking): FGCell, and SimPD (or its p) per cut-off."""
    parts = []
    for dataset, agg in aggregates.items():
        value = "SimPD_p" if "SimPD_p" in agg else "SimPD"
        wide = agg.pivot_table(index=["ModelForm", "Score", "Ranking"], columns="TopN", values=value)
        wide = wide[[c for c in ORDER["TopN"] if c in wide.columns]]
        wide.columns = [f"{dataset}_{value}_{c}" for c in wide.columns]
        fg = agg.drop_duplicates(["ModelForm", "Score", "Ranking"]).set_index(["ModelForm", "Score", "Ranking"])
        parts.append(pd.concat([fg[["FGCell"]].add_prefix(f"{dataset}_"), wide], axis=1))
    return _sorted(pd.concat(parts, axis=1).reset_index(), ["ModelForm", "Score", "Ranking"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-root", type=Path, default=HERE / "results" / "scgpt")
    p.add_argument("--with-attention", action="store_true", help="Keep the attention rows.")
    p.add_argument("--with-baseline", action="store_true",
                   help="Add simpd_baseline.py's comparison with random gene sets (p, BH q).")
    p.add_argument("--macrotable", default="macrotable", help="Macrotable folder under --results-root.")
    p.add_argument("--tables-dir", default="tables", help="Output folder under --results-root.")
    p.add_argument("--scores", nargs="+", default=["IG"],
                   help="Scores of the tables (labels; default: IG; also IG±, rollout, rollout±, gradrollout, gradrollout±).")
    args = p.parse_args()

    table = pd.read_csv(args.results_root / args.macrotable / "macrotable.csv")
    table = table[table["Model"] == "scGPT"]
    keep = set(args.scores) | ({"attention"} if args.with_attention else set())
    table = table[table["Score"].isin(keep)]
    missing = [c for c in VALUES if c not in table.columns]
    if missing:
        raise SystemExit(f"macrotable.csv has no {missing}: build it with --modules similarity faithfulness")
    out_dir = args.results_root / args.tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates = {}
    for dataset, ds in table.groupby("Dataset"):
        tag = str(dataset).lower()
        per_ct = _baseline(args.results_root, str(dataset), "methods") if args.with_baseline else None
        if per_ct is not None:
            ds = ds.merge(per_ct, on=["CellType"] + ROWS, how="left")
            ds["SimPD_q"] = _bh(ds["SimPD_p"])
        values = VALUES + (BASELINE if per_ct is not None else [])
        for cell_type, ct in ds.groupby("CellType"):
            _sorted(ct[ROWS + values], ROWS).to_csv(out_dir / f"{tag}_{cell_type}.csv", index=False)
        agg = ds.groupby(ROWS)[VALUES].mean().reset_index()
        agg["n_cell_types"] = ds.groupby(ROWS)["CellType"].nunique().to_numpy()
        # The mean over the cell types against the null of that mean (not the mean of the z)
        mean_null = _baseline(args.results_root, str(dataset), "aggregate") if args.with_baseline else None
        if mean_null is not None:
            agg = agg.merge(mean_null, on=ROWS, how="left")
            agg.insert(agg.columns.get_loc("SimPD_p") + 1, "SimPD_q", _bh(agg["SimPD_p"]))
        agg = _sorted(agg, ROWS)
        agg.to_csv(out_dir / f"{tag}_aggregate.csv", index=False)
        aggregates[str(dataset)] = agg

        # FGCell depends only on the form's per-cell explanation (IG for the IG scores, rollout
        # for the rollout ones; on the cells of one condition for the cases / controls rankings)
        explanation = ds.apply(lambda r: "per-cell " + next((m for m in ("gradrollout", "rollout")
                                                             if r["Score"].startswith(m)), "IG")
                               + (f" ({r['Ranking']})" if r["Ranking"] in ("cases", "controls") else ""), axis=1)
        fg = ds.assign(Score=explanation).drop_duplicates(["ModelForm", "Score", "CellType"]) \
            .pivot_table(index=["ModelForm", "Score"], columns="CellType", values="FGCell")
        fg["mean"] = fg.mean(axis=1)
        fg = fg.reset_index().rename(columns={"Score": "Explanation"})
        fg = fg.assign(_form=fg["ModelForm"].map({v: i for i, v in enumerate(ORDER["ModelForm"])})) \
            .sort_values("_form", kind="stable").drop(columns="_form")
        fg.to_csv(out_dir / f"{tag}_fgcell.csv", index=False)

        print(f"\n# Dataset {dataset}: mean over cell types", flush=True)
        print(agg.round(3).to_string(index=False), flush=True)
        print(f"\n# Dataset {dataset}: FGCell per cell type", flush=True)
        print(fg.round(3).to_string(index=False), flush=True)

    both = _fgcell_vs_simpd(aggregates)
    both.to_csv(out_dir / "fgcell_vs_simpd.csv", index=False)
    print("\n# FGCell against SimPD (its p against chance with --with-baseline) per form, score and ranking",
          flush=True)
    print(both.round(2).to_string(index=False), flush=True)
    print(f"\ntables in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
