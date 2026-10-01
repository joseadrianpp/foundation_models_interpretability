"""
OLS residualization of scGPT zero-shot attention against gene-level features.

For each cell type:
  1. Recompute the same signed PD-vs-control attention score used by the
     existing attention task.
  2. Compute gene-level univariate features from the original h5ad X:
       - mean expression
       - expression variance
       - dropout rate (fraction of cells with X == 0)
  3. Fit, across genes, the OLS model
       Attention_g = beta_0 + beta_1*expression_g
                              + beta_2*variance_g
                              + beta_3*dropout_g + epsilon_g
  4. Save the fitted value and residual epsilon_g for every gene.

The residual is the part of the signed attention score not explained by the
three supplied univariate gene features.

Only the scGPT forward pass runs on `device`; it must be CUDA because the
model uses the flash-attention backend (fp16/bf16 only), and it matches the
autocast forward pass of the attention task so the residualized score is the
same pooled attention. The OLS fit and selectors are numpy (CPU). Dynamic
top-N selectors are also applied to the residual attention scores, while
`ols.csv` retains all genes ranked by the residual signal. The donor
permutation selector only runs with permutation=True.
"""

from pathlib import Path
from typing import Dict, Tuple

import csv
import json

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from scipy.stats import f as f_dist

from pipeline import config as C
from pipeline.steps import _step7_common as common
from pipeline.steps import attention_cssi_leiden as cssi
from pipeline.steps import attention_dynamic_selection as dynamic
from pipeline.utils.scgpt_model import load_model_configs, load_vocab


# BH over ~1000 genes needs >= m/alpha = 20k permutations for a single gene to
# reach q < 0.05 (min q = m / (n_perm + 1)); 50k gives margin (min q ~ 0.02).
DYNAMIC_PERMUTATIONS = dynamic.PERMUTATION_N
DYNAMIC_PERMUTATION_ALPHA = 0.05
DYNAMIC_PERMUTATION_EFFECT_SIZE = dynamic.PERMUTATION_EFFECT_SIZE


def _cell_type_subsets(adata: ad.AnnData):
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    for ct in sorted(values.unique()):
        yield ct, common._slug_cell_type(ct), adata[(values == ct).to_numpy()].copy()


def _gene_features(adata: ad.AnnData) -> Dict[str, Dict[str, float]]:
    """Calculate mean expression, variance and dropout directly from adata.X."""
    X = adata.X
    n_cells = X.shape[0]
    if n_cells == 0:
        raise ValueError("Cannot calculate gene features for an empty cell type.")

    if issparse(X):
        X = X.tocsr()
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        variance = np.maximum(mean_sq - mean * mean, 0.0)
        nonzero = np.asarray(X.getnnz(axis=0)).ravel()
        dropout = 1.0 - nonzero / float(n_cells)
    else:
        X = np.asarray(X)
        mean = np.nanmean(X, axis=0)
        variance = np.nanvar(X, axis=0)
        dropout = np.mean(X == 0, axis=0)

    genes = [str(g) for g in adata.var.index]
    return {
        gene: {
            "expression_mean": float(mean[i]),
            "expression_variance": float(variance[i]),
            "dropout_rate": float(dropout[i]),
        }
        for i, gene in enumerate(genes)
    }


def _ols_residualize(
    genes,
    attention_scores: Dict[str, float],
    features: Dict[str, Dict[str, float]],
) -> Tuple[list, dict]:
    """Fit OLS across genes and return all-gene rows plus model metadata."""
    rows = []
    for gene in genes:
        gene = str(gene)
        rows.append(
            {
                "gene": gene,
                "attention_score": float(attention_scores.get(gene, 0.0)),
                **features[gene],
            }
        )

    y = np.asarray([r["attention_score"] for r in rows], dtype=float)
    feature_names = [
        "expression_mean",
        "expression_variance",
        "dropout_rate",
    ]
    X_raw = np.asarray(
        [[r[name] for name in feature_names] for r in rows],
        dtype=float,
    )

    finite = np.isfinite(y) & np.isfinite(X_raw).all(axis=1)
    if finite.sum() <= len(feature_names) + 1:
        raise ValueError(
            f"Not enough finite genes for OLS: {int(finite.sum())} available."
        )

    X_fit_raw = X_raw[finite]
    y_fit = y[finite]

    # Standardization is only for numerical conditioning. With an intercept
    # and no regularization, it does not change fitted values or residuals.
    means = X_fit_raw.mean(axis=0)
    stds = X_fit_raw.std(axis=0)
    active = stds > 0

    Z = np.zeros_like(X_fit_raw)
    Z[:, active] = (X_fit_raw[:, active] - means[active]) / stds[active]

    design = np.column_stack([np.ones(Z.shape[0]), Z])
    beta, *_ = np.linalg.lstsq(design, y_fit, rcond=None)

    predicted = np.full_like(y, np.nan, dtype=float)
    Z_all = np.zeros_like(X_raw)
    Z_all[:, active] = (X_raw[:, active] - means[active]) / stds[active]
    predicted[finite] = np.column_stack(
        [np.ones(int(finite.sum())), Z_all[finite]]
    ) @ beta

    residual = y - predicted

    ss_res = float(np.sum((y_fit - design @ beta) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_fit = int(finite.sum())
    p_predictors = len(feature_names)
    df_resid = n_fit - p_predictors - 1
    if np.isfinite(r2) and df_resid > 0 and 0.0 <= r2 < 1.0:
        f_stat = (r2 / p_predictors) / ((1.0 - r2) / df_resid)
        f_pvalue = float(f_dist.sf(f_stat, p_predictors, df_resid))
    else:
        f_stat = float("nan")
        f_pvalue = float("nan")

    for i, row in enumerate(rows):
        row["predicted_attention"] = float(predicted[i])
        row["residual_attention"] = float(residual[i])

    valid_residuals = [
        r["residual_attention"]
        for r in rows
        if np.isfinite(r["residual_attention"])
    ]
    resid_mean = float(np.mean(valid_residuals))
    resid_std = float(np.std(valid_residuals))

    for row in rows:
        if not np.isfinite(row["residual_attention"]):
            row["residual_zscore"] = float("nan")
        elif resid_std > 0:
            row["residual_zscore"] = (
                row["residual_attention"] - resid_mean
            ) / resid_std
        else:
            row["residual_zscore"] = 0.0

    # Sort the detailed audit table by residual attention (rank 1 = largest residual).
    rows.sort(
        key=lambda r: (
            -r["residual_attention"]
            if np.isfinite(r["residual_attention"])
            else float("inf")
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    metadata = {
        "n_genes": len(rows),
        "n_genes_used": n_fit,
        "predictors": feature_names,
        "standardized_predictors": True,
        "intercept": float(beta[0]),
        "coefficients_standardized": {
            name: float(beta[i + 1]) if active[i] else 0.0
            for i, name in enumerate(feature_names)
        },
        "r_squared": r2,
        "f_statistic": float(f_stat),
        "f_p_value": f_pvalue,
        "f_test_df_model": p_predictors,
        "f_test_df_residual": df_resid,
    }
    return rows, metadata


def _write_csv(rows: list, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "gene",
        "attention_score",
        "predicted_attention",
        "residual_attention",
        "residual_zscore",
        "expression_mean",
        "expression_variance",
        "dropout_rate",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _write_ols_score_csv(rows: list, out_csv: Path) -> None:
    """Write the all-gene residualized score in the standard ranking format."""
    ranked = sorted(
        (
            (str(row["gene"]), float(row["residual_attention"]))
            for row in rows
            if np.isfinite(row["residual_attention"])
        ),
        key=lambda item: (-item[1], item[0]),
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "gene", "score"])
        for rank, (gene, score) in enumerate(ranked, start=1):
            writer.writerow([rank, gene, score])


def task_attention_residualized_ols(
    adata: ad.AnnData,
    results_dir: Path,
    device: torch.device,
    donor_col=None,
    permutation: bool = False,
) -> None:
    """Run OLS residualization for every cell type of `adata` and save all genes."""
    if device.type != "cuda":
        raise ValueError(
            "attention_residualized_ols needs a CUDA device for the scGPT "
            "forward pass (flash-attention backend is fp16/bf16 GPU-only)."
        )

    vocab, model_configs = load_vocab(), load_model_configs()
    model = common._load_zero_shot_model(vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets(adata):
        print(f"\n# attention_residualized_ols | {ct} | {device} #", flush=True)

        # This deliberately uses the exact same attention computation as the
        # existing attention_zero_shot task. One forward pass feeds both the
        # cell-level condition means and, with --permutation, the donor-level matrix.
        pt = common._tokenise(sub, vocab)
        labels = pt["condition_labels"].numpy()
        if not (np.any(labels == 0) and np.any(labels == 1)):
            print(
                f"  skipped: ground-truth counts: control={int(np.sum(labels == 0))}, "
                f"pd={int(np.sum(labels == 1))}",
                flush=True,
            )
            continue

        cache = cssi.collect_scgpt(model, pt, vocab, model_configs["nheads"], device)
        attention_scores = common._gene_scores(
            cssi.condition_means(cache),
            lambda mean_pd, mean_ctrl: mean_pd - mean_ctrl,
        )

        # Same gene universe as the attention task: genes present in the
        # scGPT vocabulary and in this cell-type AnnData.
        in_vocab = np.array([g in vocab for g in sub.var.index])
        genes = [str(g) for g in sub.var.index[in_vocab]]
        sub_for_features = sub[:, in_vocab].copy()

        features = _gene_features(sub_for_features)
        rows, metadata = _ols_residualize(
            genes,
            attention_scores,
            features,
        )

        out_dir = results_dir / "attention_residualized_ols_scgpt" / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the detailed all-gene audit output.
        out_csv = out_dir / "residualized_genes.csv"
        _write_csv(rows, out_csv)

        # New standard-format OLS output: all genes ranked by residual signal.
        _write_ols_score_csv(rows, out_dir / "csv" / "ols.csv")

        # Also provide the same dynamic selectors on residualized attention.
        residual_scores = {
            str(row["gene"]): float(row["residual_attention"])
            for row in rows
            if np.isfinite(row["residual_attention"])
        }

        donor_residual_data = None
        if permutation:
            donor_column = dynamic.resolve_donor_column(sub, donor_col)
            donor_ids = sub.obs[donor_column].astype(str).to_numpy()
            print(
                f"  donor column: {donor_column} | {np.unique(donor_ids).size} donors",
                flush=True,
            )
            donor_data = dynamic.donor_attention_data(cache, donor_ids)

            # Residualize each donor's gene-attention vector with the same fitted
            # gene-level OLS prediction. This is a fixed per-gene offset, so the
            # donor-level Cohen's d is evaluated on the residualized values while
            # preserving the current OLS definition.
            predicted_by_gene = {
                str(row["gene"]): float(row["predicted_attention"])
                for row in rows
                if np.isfinite(row["predicted_attention"])
            }
            predicted_vector = np.asarray(
                [predicted_by_gene.get(g, np.nan) for g in donor_data["genes"]],
                dtype=float,
            )
            donor_residual_scores = donor_data["scores"] - predicted_vector[None, :]
            donor_residual_data = {**donor_data, "scores": donor_residual_scores}

        selection_summary = dynamic.write_selection_family(
            residual_scores,
            out_dir / "csv",
            donor_data=donor_residual_data,
            cssi_mode=False,
            random_state=42,
            permutation_n=DYNAMIC_PERMUTATIONS,
            permutation_alpha=DYNAMIC_PERMUTATION_ALPHA,
            permutation_effect_size=DYNAMIC_PERMUTATION_EFFECT_SIZE,
        )

        with open(out_dir / "ols_summary.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"  wrote {out_csv} ({len(rows)} genes; "
            f"R^2={metadata['r_squared']:.4f}; "
            f"F={metadata['f_statistic']:.3f}; "
            f"p={metadata['f_p_value']:.3e})",
            flush=True,
        )
        print(
            "  residual dynamic N: "
            + ", ".join(
                f"{method}={info['n_selected']}"
                for method, info in selection_summary.items()
            ),
            flush=True,
        )

    del model
