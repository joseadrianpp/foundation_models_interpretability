"""Context of each experiment, read from the model and OLS tasks' metadata.

  nCells             cells of the cell type the score was computed on
  nCellsScored       cells the score averages over (the correctly classified
                     ones for the LoRA scores, every cell for zero-shot)
  CSSIStrata         Leiden strata of the cell embedding (CSSI rows only)
  CSSIValidStrata    strata with >= 20 scored cells per condition, the ones
                     CSSI takes its maximum over; 0 means the CSSI score
                     fell back to the pooled one (CSSI rows only)
  OLSR2              R^2 of the OLS fit of the pooled score on expression
                     mean, variance and dropout (residual rows only)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...postprocess.gene_sets import GENE_LIST_PATHS, METHOD_META

COLUMNS = ["nCells", "nCellsScored", "CSSIStrata", "CSSIValidStrata", "OLSR2"]


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def compute(experiments: pd.DataFrame, settings) -> pd.DataFrame:
    rows = []
    for e in experiments.itertuples():
        # The model or OLS task folder of the method, e.g. ig_lora_scgpt/<ct>
        folder = e.results_dir / Path(GENE_LIST_PATHS[e.method]["top30"]).parts[0] / e.cell_type
        ols = _json(folder / "ols_summary.json")
        # OLS rows take the cells from the task whose pooled score they residualize
        score_folder = e.results_dir / Path(ols["attention_source"]).parts[0] / e.cell_type if ols else folder
        meta = _json(score_folder / "cssi_metadata.json") or {}
        is_cssi = METHOD_META[e.method]["ranking"] == "cssi"
        rows.append({
            "nCells": meta.get("n_cells", np.nan),
            "nCellsScored": meta.get("n_cells_scored", meta.get("n_cells", np.nan)),
            "CSSIStrata": meta.get("n_clusters", np.nan) if is_cssi else np.nan,
            "CSSIValidStrata": meta.get("valid_strata", np.nan) if is_cssi else np.nan,
            "OLSR2": ols["r_squared"] if ols else np.nan,
        })
    df = pd.DataFrame(rows, index=experiments.index, columns=COLUMNS)
    return df.astype({c: "Int64" for c in ["nCells", "nCellsScored", "CSSIStrata", "CSSIValidStrata"]})
