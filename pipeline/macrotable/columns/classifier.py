"""Random-forest accuracy of the experiment's genes at telling PD from control cells.

  Accuracy, BalancedAccuracy  mean over the cross-validation folds
  AUC                         mean ROC AUC over the folds (PD probability)
  MajorityAccuracy            accuracy of always predicting the majority class
                              of each test fold: the reference Accuracy must beat
  nFeaturesRF                 genes of the list used as features

The forest weighs both classes equally (class_weight="balanced"): in Dataset
B ~80% of the cells are PD, and an unweighted forest mostly predicts PD.

Features: the rank pseudo-expression of each selected gene in the cell's
Geneformer tokens, v = 1 - (p - 1) / (n - 1) for the gene at position p of
the cell's n genes and 0 when absent (as in the Geneformer OLS features).
Every model uses it, so accuracies compare across models (scGPT and scCello
genes are matched to Geneformer's tokens by symbol). Genes without a
Geneformer token are left out (nFeaturesRF).

Cells: at most settings.rf_max_cells random cells (seeded) per dataset and
cell type, all splits; "A+B" pools the cells of both datasets. Folds: by
donor (no donor on both sides), the donors of each condition spread over the
folds, the same cells and folds for every experiment of a cell type. The Geneformer datasets have no
donor column; their cells are in the same order as the scGPT h5ad files,
which do (checked on cell type and diagnosis).

Results are cached in <results>/macrotable/parts/classifier_cache.csv by
experiment, gene list and RF settings: only new or changed rows are fitted.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ... import config as C
from ...utils.progress import progress
from ...utils.splits import labels_to_int
from ..experiments import KEY, genes_hash

METRICS = ["Accuracy", "BalancedAccuracy", "AUC", "MajorityAccuracy", "nFeaturesRF"]
DONOR_COL = "donor"
FEATURES = "geneformer_rank_v1"


def _settings_tag(settings) -> str:
    return f"{FEATURES}|cells={settings.rf_max_cells}|trees={settings.rf_trees}|" \
           f"folds={settings.rf_folds}:donor-stratified|seed={settings.seed}|weights=balanced"


class CellTypeData:
    """Pseudo-expression, labels, donors and CV folds of one (dataset, cell type)."""

    def __init__(self, dataset: str, cell_type: str, settings):
        import run_tasks as rt
        from datasets import concatenate_datasets, load_from_disk
        from ...steps._step7_common import _slug_cell_type
        from ...utils.geneformer_model import load_gene_names, load_token_dictionary

        special = {v for k, v in load_token_dictionary(C.GENEFORMER_TOKEN_DICT).items() if k.startswith("<")}
        input_ids, labels, donors, self.sym_to_ids = [], [], [], {}
        # "A+B" pools the cells of A and B
        for d in dataset.split("+"):
            data_dir = rt.DATASET_DIRS[d]
            obs = pd.concat([ad.read_h5ad(data_dir / p, backed="r").obs for p in rt.SCGPT_H5ADS])
            ds = concatenate_datasets([load_from_disk(str(data_dir / p)) for p in rt.GENEFORMER_DATASETS])
            gf_cell_types = np.array(ds[C.CELL_TYPE_COL])
            if len(obs) != len(gf_cell_types) or (obs[C.CELL_TYPE_COL].astype(str).to_numpy() != gf_cell_types).any():
                raise ValueError(f"dataset {d}: the scGPT and Geneformer cells are not in the same order")
            idx = np.where(np.array([_slug_cell_type(c) for c in gf_cell_types]) == cell_type)[0]
            input_ids += [np.asarray(x) for x in ds.select(idx)["input_ids"]]
            labels.append(labels_to_int(obs[C.LABEL_COL].to_numpy()[idx]))
            donors.append(f"{d}:" + obs[DONOR_COL].astype(str).to_numpy()[idx].astype(object))
            for tid, sym in load_gene_names(data_dir / C.GENEFORMER_GENE_NAMES).items():
                self.sym_to_ids.setdefault(str(sym), set()).add(int(tid))
        labels, donors = np.concatenate(labels), np.concatenate(donors)

        # Seeded subset, the same for every experiment of this cell type
        keep = np.arange(len(labels))
        if len(keep) > settings.rf_max_cells:
            keep = np.sort(np.random.default_rng(settings.seed).choice(keep, settings.rf_max_cells, replace=False))
        self.labels, self.donors = labels[keep], donors[keep]
        self.X = self._rank_matrix([input_ids[i] for i in keep], special)
        self.folds = self._donor_folds(settings)

    def _donor_folds(self, settings) -> list:
        """(train, test) cell indices, the donors of each condition spread over the folds.

        Stratifying donors, not cells, puts control and PD donors in every test
        fold even when a few donors hold most cells (StratifiedGroupKFold may not).
        """
        donors, first = np.unique(self.donors, return_index=True)
        splitter = StratifiedKFold(n_splits=settings.rf_folds, shuffle=True, random_state=settings.seed)
        folds = []
        for _, test_donors in splitter.split(donors, self.labels[first]):
            test = np.isin(self.donors, donors[test_donors])
            folds.append((np.where(~test)[0], np.where(test)[0]))
        return folds

    @staticmethod
    def _rank_matrix(input_ids: list, special: set) -> sparse.csc_matrix:
        """Cells x token ids, the rank pseudo-expression of every gene token."""
        special = np.array(sorted(special))
        cols, vals, ptr = [], [], [0]
        for ids in input_ids:
            ids = ids[~np.isin(ids, special)]
            n = ids.size
            cols.append(ids)
            vals.append(1.0 - np.arange(n) / (n - 1) if n > 1 else np.ones(n))
            ptr.append(ptr[-1] + n)
        cols = np.concatenate(cols)
        m = sparse.csr_matrix((np.concatenate(vals).astype(np.float32), cols, ptr),
                              shape=(len(input_ids), int(cols.max()) + 1))
        return m.tocsc()

    def features(self, genes: list) -> np.ndarray:
        """One column per gene with a Geneformer token (tokens of one symbol summed)."""
        columns = []
        for g in genes:
            ids = [i for i in self.sym_to_ids.get(g, ()) if i < self.X.shape[1]]
            if ids:
                columns.append(np.asarray(self.X[:, ids].sum(axis=1)).ravel())
        return np.column_stack(columns) if columns else np.zeros((len(self.labels), 0))


def _fit(data: CellTypeData, genes: list, settings) -> dict:
    X = data.features(genes)
    if X.shape[1] == 0:
        return {"Accuracy": np.nan, "BalancedAccuracy": np.nan, "AUC": np.nan,
                "MajorityAccuracy": _majority_accuracy(data), "nFeaturesRF": 0}
    acc, bacc, auc = [], [], []
    for train, test in data.folds:
        rf = RandomForestClassifier(n_estimators=settings.rf_trees, n_jobs=settings.n_jobs,
                                    random_state=settings.seed, class_weight="balanced")
        rf.fit(X[train], data.labels[train])
        pred = rf.predict(X[test])
        acc.append(accuracy_score(data.labels[test], pred))
        bacc.append(balanced_accuracy_score(data.labels[test], pred))
        auc.append(roc_auc_score(data.labels[test], rf.predict_proba(X[test])[:, 1]))
    return {"Accuracy": float(np.mean(acc)), "BalancedAccuracy": float(np.mean(bacc)),
            "AUC": float(np.mean(auc)), "MajorityAccuracy": _majority_accuracy(data), "nFeaturesRF": X.shape[1]}


def _majority_accuracy(data: CellTypeData) -> float:
    """Mean over the folds of the accuracy of predicting the test fold's majority class."""
    return float(np.mean([max(data.labels[te].mean(), 1 - data.labels[te].mean()) for _, te in data.folds]))


def compute(experiments: pd.DataFrame, settings) -> pd.DataFrame:
    cache_path = Path(settings.results_root) / "macrotable" / "parts" / "classifier_cache.csv"
    tag = _settings_tag(settings)
    cache = (pd.read_csv(cache_path, dtype={c: str for c in KEY + ["genes_hash", "settings"]})
             if cache_path.exists() else pd.DataFrame(columns=KEY + ["genes_hash", "settings"] + METRICS))
    if "Score" not in cache.columns:
        # Cached before the Score key column: every LoRA scGPT row was IG, every other one attention
        lora_ig = (cache["Model"] == "scGPT") & (cache["ModelForm"] == "LoRA")
        cache.insert(KEY.index("Score"), "Score", np.where(lora_ig, "IG", "attention"))
    cached = {tuple(r[KEY + ["genes_hash", "settings"]]): {m: r.get(m, np.nan) for m in METRICS} for _, r in cache.iterrows()}

    hashes = [genes_hash(g) for g in experiments["genes"]]
    todo = [i for i, (k, h) in enumerate(zip(experiments.index, hashes)) if (*k, h, tag) not in cached]
    print(f"  [classifier] {len(experiments) - len(todo)} cached, {len(todo)} to fit ({tag})", flush=True)

    new_rows = []
    # One (dataset, cell type) at a time: its cells and folds are loaded once
    todo_df = experiments.iloc[todo].assign(ghash=[hashes[i] for i in todo])
    for (dataset, cell_type), group in todo_df.groupby(["Dataset", "cell_type"], sort=False):
        data = CellTypeData(dataset, cell_type, settings)
        for e in progress(list(group.itertuples()), desc=f"RF {dataset} {cell_type}", unit="experiment"):
            result = _fit(data, e.genes, settings)
            cached[(*e.Index, e.ghash, tag)] = result
            new_rows.append({**dict(zip(KEY, e.Index)), "genes_hash": e.ghash, "settings": tag, **result})
        if new_rows:
            new = pd.DataFrame(new_rows)
            cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_csv(cache_path, index=False)
            new_rows = []

    rows = [cached[(*k, h, tag)] for k, h in zip(experiments.index, hashes)]
    return pd.DataFrame(rows, index=experiments.index, columns=METRICS)
