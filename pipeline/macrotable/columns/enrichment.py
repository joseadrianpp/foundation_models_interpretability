"""Enrichment statistics of the experiment's genes, per GO sub-ontology and all GO together.

For each group G (GOALL = GO:BP + GO:CC + GO:MF terms pooled, then CC, BP, MF),
over the significant g:Profiler terms of that group (enrichment task):

  <G>IC              mean information content, -log2(term_size / domain_size)
  <G>FunctAbundance  mean functional abundance, intersection_size / term_size
  <G>MIS             mean intersection size
  <G>TermCount       number of significant terms

The per-term metrics are the ones of postprocess/enrichment_summary.py. With
no significant term in a group the means are NaN and TermCount is 0; without
an enrichment CSV for the experiment every column is NaN.
"""

import numpy as np
import pandas as pd

from ...postprocess.enrichment_summary import _load_terms
from ...postprocess.gene_sets import output_dir

GROUPS = {"GOALL": ["GO:BP", "GO:CC", "GO:MF"], "CC": ["GO:CC"], "BP": ["GO:BP"], "MF": ["GO:MF"]}
STATS = {"IC": "information_content", "FunctAbundance": "functional_abundance", "MIS": "intersection_size"}


def _group_stats(terms: pd.DataFrame) -> dict:
    row = {}
    for group, sources in GROUPS.items():
        sub = terms[terms["source"].isin(sources)] if len(terms) else terms
        for name, metric in STATS.items():
            row[f"{group}{name}"] = float(sub[metric].mean()) if len(sub) else np.nan
        row[f"{group}TermCount"] = len(sub)
    return row


def compute(experiments: pd.DataFrame, settings) -> pd.DataFrame:
    rows = []
    for e in experiments.itertuples():
        enrichment_dir = output_dir(e.results_dir, "enrichment", e.selection)
        if not (enrichment_dir / e.method / f"{e.cell_type}.csv").exists():
            rows.append({f"{g}{s}": np.nan for g in GROUPS for s in [*STATS, "TermCount"]})
            continue
        rows.append(_group_stats(_load_terms(enrichment_dir, e.method, e.cell_type)))
    # Integer counts that can still be missing
    return pd.DataFrame(rows, index=experiments.index).astype({f"{g}TermCount": "Int64" for g in GROUPS})
