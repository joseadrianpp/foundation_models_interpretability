"""Experiment table ("macrotable"): one row per experiment, one column group per module.

An experiment is one gene list: ranking x top-N selection x model x model form
x dataset x cell type (experiments.py builds the rows from the gene lists
registered in postprocess/gene_sets.py). Each module of columns/ computes one
group of columns for any set of experiments, with the same interface:

    compute(experiments: pd.DataFrame, settings: Settings) -> pd.DataFrame

returning one row per experiment, indexed like `experiments` (by KEY).
build.py runs the requested modules, saves each part to
results/macrotable/parts/<module>.csv and joins every part on disk into
results/macrotable/macrotable.csv, so one module can be recomputed alone.

A new column group is a new module registered in build.MODULES; a new
dataset, model or model form only needs its gene lists registered in
gene_sets.py.
"""
