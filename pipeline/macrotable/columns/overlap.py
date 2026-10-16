"""Selected genes and their overlap with the three Parkinson's panels.

  genes, nGenes                 the experiment's gene list (highest score first)
  genes<P>, nGenes<P>, pGenes<P> genes shared with panel P (GWAS, GE, GO), how
                                many, and the hypergeometric p-value of that
                                overlap (P(X >= k))

The p-value draws the selection from the genes the model could rank in that
cell type (every gene of its zero-shot attention score: every expressed
gene of the h5ad for scGPT, every gene seen for Geneformer and scCello), so
overlaps are comparable between models with different gene universes. The panels are the reference gene
sets saved by the gwas_similarity task (gwas_similarity/refs/).
"""

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Column suffix -> reference gene set of gwas_similarity/refs/
PANELS = {"GWAS": "parkinson", "GE": "panelapp", "GO": "parkinsonsuk_ucl"}


def _universe(results_dir, model: str, cell_type: str) -> set:
    path = results_dir / f"attention_zero_shot_{model}" / cell_type / "csv" / "all_genes.csv"
    return set(pd.read_csv(path)["gene"].astype(str))


def _panel(results_dir, ref: str) -> set:
    return set(pd.read_csv(results_dir / "gwas_similarity" / "refs" / f"{ref}_genes.csv")["gene"].astype(str))


def compute(experiments: pd.DataFrame, settings) -> pd.DataFrame:
    universes, panels, rows = {}, {}, []
    for e in experiments.itertuples():
        u_key = (e.results_dir, e.model, e.cell_type)
        if u_key not in universes:
            universes[u_key] = _universe(*u_key)
        universe = universes[u_key]
        selected = set(e.genes)
        row = {"genes": ";".join(e.genes), "nGenes": len(e.genes)}
        for suffix, ref in PANELS.items():
            if (e.results_dir, ref) not in panels:
                panels[(e.results_dir, ref)] = _panel(e.results_dir, ref)
            panel = panels[(e.results_dir, ref)]
            shared = [g for g in e.genes if g in panel]
            # Population: the model's genes; successes: panel genes among them
            n_pop, n_success = len(universe | selected), len(panel & (universe | selected))
            row[f"genes{suffix}"] = ";".join(shared)
            row[f"nGenes{suffix}"] = len(shared)
            row[f"pGenes{suffix}"] = float(hypergeom.sf(len(shared) - 1, n_pop, n_success, len(selected))) \
                if n_success else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=experiments.index)
