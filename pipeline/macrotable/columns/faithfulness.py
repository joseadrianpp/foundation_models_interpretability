"""Faithfulness gain (FG) of the experiment's ranking and of its model's per-cell IG.

From evaluate_attributions.py --curves --rankings, in
<results>/dataset_<x>/attribution_eval/<model>_<form>/fg_by_cell_type.csv
(mean FG per method and cell type):
  FG       the ranking itself (its pooled / CSSI / OLS scores of score IG or
           IG±) used as every cell's attribution: Problem B, evaluated like A.
           The same for every TopN of the ranking.
  FGcell   the form's per-cell signed IG (method `ig`): Problem A, the same
           for every ranking and TopN of the form.
FG = 1 - deletion AUC / deletion AUC of a random attribution on the same
cell (0 = random, higher = better). NaN where the evaluation has not run
(attention rankings are not evaluated).
"""

import numpy as np
import pandas as pd

from ...postprocess import gene_sets as gs

# model_form of gene_sets.py -> form folder part
FORM_DIRS = {model_form: form for form, (model_form, _) in gs.FORMS.items()}


def _fg_table(results_dir, model: str, model_form: str):
    path = results_dir / "attribution_eval" / f"{model}_{FORM_DIRS[model_form]}" / "fg_by_cell_type.csv"
    return pd.read_csv(path, index_col="method") if path.exists() else None


def compute(experiments: pd.DataFrame, settings) -> pd.DataFrame:
    tables = {}
    rows = []
    for e in experiments.itertuples():
        meta = gs.METHOD_META[e.method]
        key = (e.results_dir, meta["model"], meta["model_form"])
        if key not in tables:
            tables[key] = _fg_table(*key)
        fg = tables[key]
        ranking = f"rank_{meta['score']}_{meta['ranking']}"
        lookup = lambda method: (float(fg.at[method, e.cell_type])
                                 if fg is not None and method in fg.index and e.cell_type in fg.columns else np.nan)
        rows.append({"FG": lookup(ranking) if meta["score"] != "attention" else np.nan, "FGcell": lookup("ig")})
    return pd.DataFrame(rows, index=experiments.index, columns=["FG", "FGcell"])
