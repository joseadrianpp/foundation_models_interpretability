"""GO semantic similarity of the experiment's genes to each disease reference.

Sim<R>_<O>: GO3 set-level BMA similarity (gwas_similarity task) between the
gene list and reference R, in sub-ontology O (BP, CC, MF), with the first of
settings.distances (SimRel by default); further distances add a suffix,
Sim<R>_<O>_<distance>. A reference the gwas_similarity task did not compute
yet gives NaN columns, so the table keeps the same columns.
"""

import numpy as np
import pandas as pd

from ...postprocess.gene_sets import output_dir

# Column label -> reference of gwas_similarity
REFERENCES = {"PDGWAS": "parkinson", "PDGE": "panelapp", "PDGO": "parkinsonsuk_ucl",
              "AD": "alzheimer", "Kidney": "kidney", "CAD": "cad"}
ONTOLOGIES = ["BP", "CC", "MF"]


def _columns(distances: list) -> dict:
    """{(reference, ontology, distance): column name}, in table order."""
    cols = {}
    for i, dist in enumerate(distances):
        suffix = "" if i == 0 else f"_{dist}"
        for label, ref in REFERENCES.items():
            for ont in ONTOLOGIES:
                cols[(ref, f"GO:{ont}", dist)] = f"Sim{label}_{ont}{suffix}"
    return cols


def compute(experiments: pd.DataFrame, settings) -> pd.DataFrame:
    cols = _columns(settings.distances)
    sims = {}
    for results_dir, selection in experiments[["results_dir", "selection"]].drop_duplicates().itertuples(index=False):
        path = output_dir(results_dir, "gwas_similarity", selection) / "gene_level_similarity_all.csv"
        if path.exists():
            df = pd.read_csv(path)
            for r in df.itertuples():
                sims[(results_dir, selection, r.method, r.cell_type, r.reference, r.go_source, r.distance)] = r.similarity
    rows = []
    for e in experiments.itertuples():
        rows.append({col: sims.get((e.results_dir, e.selection, e.method, e.cell_type, *k), np.nan)
                     for k, col in cols.items()})
    return pd.DataFrame(rows, index=experiments.index, columns=list(cols.values()))
