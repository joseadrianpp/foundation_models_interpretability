"""Faithfulness of the experiment's model: FGCell of its per-cell explanation.

From evaluate_attributions.py, in
<results>/dataset_<x>/attribution_eval/<model>_<form>/fg_by_cell_type.csv
(mean faithfulness gain per method and cell type):
  FGCell   the form's per-cell explanation the score comes from (method `ig`
           for IG and IG±, `rollout` / `gradrollout` for the rollout ones),
           each cell's genes removed in the order of its own attribution:
           the same for every ranking and TopN of the form and score, except
           the cases / controls rankings, whose FGCell (`ig_cases`,
           `ig_controls`) is the mean over the cells of that condition only.
FGCell = 1 - deletion AUC / deletion AUC of a random attribution on the same
cell (0 = random, higher = better; above 1 is possible for an encoder, whose
output can fall below that of the baseline). NaN where the evaluation has not
run. The FG of the population rankings (every cell explained by the pooled
ranking) was dropped on 2026-10-31: it is near chance for the classifiers and
does not tell whether the explanation reflects each cell's decision.
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
        cell_method = next((m for m in ("gradrollout", "rollout") if meta["score"].startswith(m)), "ig")
        if meta["ranking"] in ("cases", "controls"):
            # The per-cell IG on the cells of that condition only
            cell_method = f"{cell_method}_{meta['ranking']}"
        found = fg is not None and cell_method in fg.index and e.cell_type in fg.columns
        rows.append({"FGCell": float(fg.at[cell_method, e.cell_type]) if found else np.nan})
    return pd.DataFrame(rows, index=experiments.index, columns=["FGCell"])
