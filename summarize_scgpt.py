#!/usr/bin/env python
"""Result tables of the scGPT study: FG and SimPD per form, score, ranking and cut-off.

Reads <results-root>/macrotable/macrotable.csv, built with
`python -m pipeline.macrotable.build --results-dir <results-root> --models scgpt
--modules similarity faithfulness`, and writes to <results-root>/tables/:

  <dataset>_<cell type>.csv   one row per (form, score, ranking, TopN): SimPD
                              of the gene set (Problem C), FG of the ranking
                              (Problem B) and FG of the form's per-cell IG
                              (Problem A)
  <dataset>_aggregate.csv     the same rows, mean over the four cell types
  <dataset>_fg.csv            FG of every ranking and of the per-cell IG
                              (rows) per cell type, and their mean (columns)
  fg_vs_simpd.csv             one row per (form, score, ranking): FG of the
                              per-cell IG and of the ranking, and the SimPD
                              z of each cut-off against chance, both datasets

With simpd_baseline.py's outputs (<results-root>/simpd_baseline/), the
SimPD rows also get its comparison with random gene sets drawn with the same
detection rates: SimPD_null (their mean), SimPD_z and SimPD_p (one-sided),
and SimPD_z_uniform (against sets of any genes); the aggregate rows, against
the null of the mean over the cell types.

The attention rows are left out (the study dropped raw attention) unless
--with-attention.

  python summarize_scgpt.py --results-root results/scgpt
"""

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROWS = ["ModelForm", "Score", "Ranking", "TopN"]
VALUES = ["SimPD", "FG", "FGCell"]
BASELINE = ["SimPD_null", "SimPD_z", "SimPD_p", "SimPD_z_uniform"]
# simpd_baseline.py's names -> the macrotable's labels
FORM_LABELS = {"zero_shot": "zero-shot", "step2_5": "Step2.5", "mlp": "zero-shot+MLP", "lora": "LoRA"}
SCORE_LABELS = {"ig": "IG", "igsigned": "IG±", "rollout": "rollout", "rolloutsigned": "rollout±",
                "gradrollout": "gradrollout", "gradrolloutsigned": "gradrollout±"}
RANKING_LABELS = {None: "pooled", "cssi_": "CSSI", "residualized_ols_": "residualized"}
CELL_TYPE_LABELS = {"astrocytes": "astrocyte", "da_neurons": "neuron", "microglia": "microglia",
                    "oligodendrocytes": "oligodendrocyte"}
TOPN_LABELS = {"top30": "top30", "z-score": "zscore", "elbow": "elbow"}
# Table order of the key values
ORDER = {"ModelForm": ["zero-shot", "Step2.5", "zero-shot+MLP", "LoRA"],
         "Score": ["IG", "IG±", "rollout", "rollout±", "gradrollout", "gradrollout±", "attention"],
         "Ranking": ["pooled", "CSSI", "residualized"],
         "TopN": ["top30", "zscore", "elbow"]}


def _sorted(df: pd.DataFrame, cols) -> pd.DataFrame:
    keys = {c: df[c].map({v: i for i, v in enumerate(ORDER[c])}) for c in cols}
    return df.assign(**{f"_{c}": k for c, k in keys.items()}).sort_values([f"_{c}" for c in cols]) \
        .drop(columns=[f"_{c}" for c in cols])


def _baseline(results_root: Path, dataset: str, part: str):
    """simpd_baseline.py's comparison with chance (part: methods or aggregate), in macrotable labels."""
    path = results_root / "simpd_baseline" / f"{dataset}_{part}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keys = df["method"].str.extract(r"^(igsigned|ig|gradrolloutsigned|gradrollout|rolloutsigned|rollout)_(cssi_|residualized_ols_)?(.*)_scgpt$")
    out = pd.DataFrame({"ModelForm": keys[2].map(FORM_LABELS), "Score": keys[0].map(SCORE_LABELS),
                        "Ranking": keys[1].map(lambda r: RANKING_LABELS[r if isinstance(r, str) else None]),
                        "TopN": df["selection"].map(TOPN_LABELS),
                        "SimPD_null": df["expr_mean"], "SimPD_z": df["expr_z"], "SimPD_p": df["expr_p"],
                        "SimPD_z_uniform": df["uniform_z"]})
    if "cell_type" in df:
        out.insert(0, "CellType", df["cell_type"].map(CELL_TYPE_LABELS))
    return out


def _fg_vs_simpd(aggregates: dict) -> pd.DataFrame:
    """One row per (form, score, ranking): FG of the per-cell IG and of the ranking, SimPD z per cut-off."""
    parts = []
    for dataset, agg in aggregates.items():
        wide = agg.pivot_table(index=["ModelForm", "Score", "Ranking"], columns="TopN",
                               values="SimPD_z" if "SimPD_z" in agg else "SimPD")[ORDER["TopN"]]
        wide.columns = [f"{dataset}_SimPD_{'z' if 'SimPD_z' in agg else 'raw'}_{c}" for c in wide.columns]
        fg = agg.drop_duplicates(["ModelForm", "Score", "Ranking"]).set_index(["ModelForm", "Score", "Ranking"])
        parts.append(pd.concat([fg[["FGCell", "FG"]].add_prefix(f"{dataset}_"), wide], axis=1))
    return _sorted(pd.concat(parts, axis=1).reset_index(), ["ModelForm", "Score", "Ranking"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-root", type=Path, default=HERE / "results" / "scgpt")
    p.add_argument("--with-attention", action="store_true", help="Keep the attention rows.")
    p.add_argument("--macrotable", default="macrotable", help="Macrotable folder under --results-root.")
    p.add_argument("--tables-dir", default="tables", help="Output folder under --results-root.")
    p.add_argument("--scores", nargs="+", default=["IG", "IG±"],
                   help="Scores of the tables (labels; default: IG IG±; also rollout, rollout±, gradrollout, gradrollout±).")
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
        per_ct = _baseline(args.results_root, str(dataset), "methods")
        if per_ct is not None:
            ds = ds.merge(per_ct, on=["CellType"] + ROWS, how="left")
        values = VALUES + (BASELINE if per_ct is not None else [])
        for cell_type, ct in ds.groupby("CellType"):
            _sorted(ct[ROWS + values], ROWS).to_csv(out_dir / f"{tag}_{cell_type}.csv", index=False)
        agg = ds.groupby(ROWS)[VALUES].mean().reset_index()
        agg["n_cell_types"] = ds.groupby(ROWS)["CellType"].nunique().to_numpy()
        # The mean over the cell types against the null of that mean (not the mean of the z)
        mean_null = _baseline(args.results_root, str(dataset), "aggregate")
        if mean_null is not None:
            agg = agg.merge(mean_null, on=ROWS, how="left")
        agg = _sorted(agg, ROWS)
        agg.to_csv(out_dir / f"{tag}_aggregate.csv", index=False)
        aggregates[str(dataset)] = agg

        # FG does not depend on the cut-off: one row per ranking, plus each form's per-cell
        # explanations (IG for the IG scores, rollout for the rollout ones)
        rank_fg = ds.drop_duplicates(["ModelForm", "Score", "Ranking", "CellType"]) \
            .pivot_table(index=["ModelForm", "Score", "Ranking"], columns="CellType", values="FG")
        explanation = ds["Score"].map(lambda s: "per-cell " + next((m for m in ("gradrollout", "rollout")
                                                                     if s.startswith(m)), "IG"))
        cell_fg = ds.assign(Score=explanation).drop_duplicates(["ModelForm", "Score", "CellType"]) \
            .pivot_table(index=["ModelForm", "Score"], columns="CellType", values="FGCell")
        cell_fg.index = pd.MultiIndex.from_tuples([(f, s, "") for f, s in cell_fg.index], names=rank_fg.index.names)
        fg = pd.concat([rank_fg, cell_fg])
        fg["mean"] = fg.mean(axis=1)
        # Forms in table order, each form's per-cell explanations first and then its rankings
        fg = fg.reset_index()
        fg = fg.assign(_form=fg["ModelForm"].map({v: i for i, v in enumerate(ORDER["ModelForm"])}),
                       _cell=~fg["Score"].str.startswith("per-cell")) \
            .sort_values(["_form", "_cell"], kind="stable").drop(columns=["_form", "_cell"])
        fg.to_csv(out_dir / f"{tag}_fg.csv", index=False)

        print(f"\n# Dataset {dataset}: mean over cell types", flush=True)
        print(agg.round(3).to_string(index=False), flush=True)
        print(f"\n# Dataset {dataset}: FG per cell type", flush=True)
        print(fg.round(3).to_string(index=False), flush=True)

    both = _fg_vs_simpd(aggregates)
    both.to_csv(out_dir / "fg_vs_simpd.csv", index=False)
    print("\n# FG against SimPD (z against chance) per form, score and ranking", flush=True)
    print(both.round(2).to_string(index=False), flush=True)
    print(f"\ntables in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
