"""Dynamic top-N selection with donor-level permutation testing.

Fixed selectors:
  * top30: 30 highest method scores.
  * z-score: score > mean + 2.5 * std.
  * elbow: Kneedle elbow on the descending score curve.

Permutation selector:
  * statistical unit = donor, never individual cells;
  * PD/control donor labels are permuted while preserving the observed number
    of PD and control donors;
  * two-sided Cohen's d is calculated gene-wise for every permutation;
  * empirical p-values are BH-adjusted to q-values;
  * a gene is selected only when q < alpha AND |d| >= effect_size;
  * selected genes are written sorted by the method's original score.

For CSSI, the statistic is the maximum absolute Cohen's d across valid Leiden
strata, with the same max-over-strata operation applied in every permutation.

The permutation selector is opt-in (run_tasks.py --permutation): without donor
data `write_selection_family` only writes the fixed selectors.
"""

from pathlib import Path
import csv
import json
from typing import Dict, List, Tuple, Optional

import numpy as np

from pipeline.steps import attention_cssi_leiden as cssi

TOP_N = 30
Z_THRESHOLD = 2.5
ELBOW_S = 1.0
PERMUTATION_N = 50000
PERMUTATION_ALPHA = 0.05
PERMUTATION_EFFECT_SIZE = 0.2
MIN_DONORS_PER_GROUP = 2
# Observed and permuted d are computed with different (but equivalent) variance
# formulas, so an exact tie (the observed donor split recurring in a
# permutation, frequent for genes seen in few donors) can differ by ~1e-13.
# Null values within this relative tolerance count as >= observed.
TIE_RTOL = 1e-9


def _ranked_items(score_map: Dict[str, float]) -> List[Tuple[str, float]]:
    items = []
    for gene, score in score_map.items():
        score = float(score)
        if np.isfinite(score):
            items.append((str(gene), score))
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def _select_top_k(items: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    return items[: max(1, min(int(k), len(items)))] if items else []


def _zscore(items: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], dict]:
    scores = np.asarray([s for _, s in items], dtype=float)
    mu = float(scores.mean())
    sigma = float(scores.std())
    threshold = mu + Z_THRESHOLD * sigma
    keep = scores > threshold
    fallback = not bool(np.any(keep))
    if fallback:
        keep[np.argmax(scores)] = True
    return [item for item, flag in zip(items, keep) if flag], {
        "mean": mu,
        "std": sigma,
        "z_threshold": Z_THRESHOLD,
        "threshold": threshold,
        "fallback_top1": fallback,
    }


def _elbow(items: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], dict]:
    if len(items) == 1:
        return items, {"k": 1, "fallback": "single_gene"}
    try:
        from kneed import KneeLocator
    except ImportError as exc:
        raise RuntimeError(
            "The elbow selector requires kneed. Install it with "
            "`pip install kneed==0.8.6`."
        ) from exc
    x = np.arange(1, len(items) + 1, dtype=float)
    y = np.asarray([s for _, s in items], dtype=float)
    locator = KneeLocator(
        x, y, S=ELBOW_S, curve="convex", direction="decreasing", online=False
    )
    if locator.knee is None:
        k = min(TOP_N, len(items))
        meta = {"k": int(k), "fallback": "top30"}
    else:
        k = max(1, min(int(round(float(locator.knee))), len(items)))
        meta = {"k": int(k), "fallback": None}
    return _select_top_k(items, k), meta


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adj = ranked * p_values.size / np.arange(1, p_values.size + 1, dtype=float)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def resolve_donor_column(adata, explicit: Optional[str] = None) -> str:
    """Return an explicit donor column or a conservative donor-like match."""
    if explicit:
        if explicit not in adata.obs.columns:
            raise ValueError(
                f"Donor column {explicit!r} not present in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return explicit

    candidates = [
        "donor", "donor_id", "donorID", "subject", "subject_id",
        "individual", "individual_id", "patient", "patient_id",
        "Brain_Bank_ID", "Brain_Bank_Path_ID", "brain_bank_id",
    ]
    present = [c for c in candidates if c in adata.obs.columns]
    if not present:
        raise ValueError(
            "Could not identify a donor column automatically. Set --donor-col. "
            f"Available adata.obs columns: {list(adata.obs.columns)}"
        )
    return present[0]


def validate_donor_labels(donor_ids: np.ndarray, labels: np.ndarray):
    donor_ids = np.asarray(donor_ids).astype(str)
    labels = np.asarray(labels, dtype=int)
    donors = np.unique(donor_ids)
    donor_labels = np.empty(len(donors), dtype=int)
    for i, donor in enumerate(donors):
        unique = np.unique(labels[donor_ids == donor])
        if unique.size != 1 or int(unique[0]) not in (0, 1):
            raise ValueError(
                f"Donor {donor!r} has non-unique/unrecognised diagnosis labels: "
                f"{unique.tolist()}"
            )
        donor_labels[i] = int(unique[0])
    return donors, donor_labels


# Donor x gene mean attention over the cells in `cell_mask`. A donor's mean is
# over all its cells (a cell without the gene adds 0); donors without cells in
# the mask get a NaN row, which the Cohen's d below skips
def _donor_gene_means(cache: cssi.AttentionCache, donor_idx: np.ndarray, n_donors: int,
                      cell_mask: Optional[np.ndarray] = None) -> np.ndarray:
    n_genes = cache.genes.size
    cells = np.ones(cache.n_cells, dtype=bool) if cell_mask is None else cell_mask
    tokens = cells[cache.token_cells]
    key = donor_idx[cache.token_cells[tokens]].astype(np.int64) * n_genes + cache.gene_idx[tokens]
    sums = np.bincount(key, weights=cache.attn[tokens], minlength=n_donors * n_genes)
    n_cells = np.bincount(donor_idx[cells], minlength=n_donors).astype(float)
    return np.divide(
        sums.reshape(n_donors, n_genes), n_cells[:, None],
        out=np.full((n_donors, n_genes), np.nan),
        where=n_cells[:, None] > 0,
    )


def donor_attention_data(cache: cssi.AttentionCache, donor_ids,
                         strata: Optional[np.ndarray] = None) -> dict:
    """Donor-level attention matrices for the permutation selector.

    Read from the same AttentionCache as the pooled/CSSI scores, so no second
    forward pass is needed. With `strata`, also one matrix per Leiden stratum
    (CSSI permutation).
    """
    donor_ids = np.asarray(donor_ids).astype(str)
    if donor_ids.size != cache.n_cells:
        raise ValueError("donor_ids must have one entry per cached cell")
    donors, donor_labels = validate_donor_labels(donor_ids, cache.labels)
    # np.unique sorts, so searchsorted maps every cell to its donor row
    donor_idx = np.searchsorted(donors, donor_ids)
    D = len(donors)

    strata_scores = {}
    if strata is not None:
        strata = np.asarray(strata, dtype=int)
        if strata.size != cache.n_cells:
            raise ValueError("strata must have one label per cell")
        for si in np.unique(strata):
            strata_scores[int(si)] = _donor_gene_means(cache, donor_idx, D, strata == si)

    return {
        "donors": donors.tolist(),
        "labels": donor_labels,
        "genes": cache.genes.tolist(),
        "scores": _donor_gene_means(cache, donor_idx, D),
        "strata": strata_scores,
    }



def _cohen_d_from_groups(X: np.ndarray, group1: np.ndarray):
    valid = np.isfinite(X)
    g1 = group1.astype(bool)[:, None] & valid
    g0 = (~group1.astype(bool))[:, None] & valid
    n1 = g1.sum(axis=0).astype(float)
    n0 = g0.sum(axis=0).astype(float)
    X0 = np.where(valid, X, 0.0)
    s1 = (g1 * X0).sum(axis=0)
    s0 = (g0 * X0).sum(axis=0)
    m1 = np.divide(s1, n1, out=np.full(X.shape[1], np.nan), where=n1 > 0)
    m0 = np.divide(s0, n0, out=np.full(X.shape[1], np.nan), where=n0 > 0)
    ss1 = (g1 * (X0 - m1[None, :]) ** 2).sum(axis=0)
    ss0 = (g0 * (X0 - m0[None, :]) ** 2).sum(axis=0)
    v1 = np.divide(ss1, n1 - 1, out=np.full(X.shape[1], np.nan), where=n1 > 1)
    v0 = np.divide(ss0, n0 - 1, out=np.full(X.shape[1], np.nan), where=n0 > 1)
    pv = np.divide(
        (n1 - 1) * v1 + (n0 - 1) * v0,
        n1 + n0 - 2,
        out=np.full(X.shape[1], np.nan),
        where=(n1 > 1) & (n0 > 1),
    )
    sd = np.sqrt(np.maximum(pv, 0.0))
    d = np.divide(
        m1 - m0,
        sd,
        out=np.full(X.shape[1], np.nan),
        where=(n1 >= MIN_DONORS_PER_GROUP) & (n0 >= MIN_DONORS_PER_GROUP) & (sd > 0),
    )
    return d, n0.astype(int), n1.astype(int)


def _permutation_labels(n_donors: int, n_pd: int, n_permutations: int, rng):
    ranks = rng.random((n_permutations, n_donors)).argsort(axis=1)
    labels = np.zeros((n_permutations, n_donors), dtype=bool)
    labels[np.arange(n_permutations)[:, None], ranks[:, :n_pd]] = True
    return labels


def _permute_matrix_d(X, donor_labels, perm_labels):
    """Return observed d and P x G null d for one donor x gene matrix."""
    observed_d, n0_obs, n1_obs = _cohen_d_from_groups(X, donor_labels == 1)
    P = perm_labels.shape[0]
    D, G = X.shape
    perm_labels = np.asarray(perm_labels, dtype=float)
    if perm_labels.shape[1] != D:
        raise ValueError("Permutation label matrix has the wrong donor dimension")
    valid = np.isfinite(X)
    X0 = np.where(valid, X, 0.0)
    V = valid.astype(float)
    sum1 = perm_labels @ X0
    n1 = perm_labels @ V
    total_sum = X0.sum(axis=0)[None, :]
    total_n = V.sum(axis=0)[None, :]
    sum0 = total_sum - sum1
    n0 = total_n - n1
    mean1 = np.divide(sum1, n1, out=np.full_like(sum1, np.nan), where=n1 > 0)
    mean0 = np.divide(sum0, n0, out=np.full_like(sum0, np.nan), where=n0 > 0)
    sq = X0 * X0
    sq1 = perm_labels @ sq
    sq0 = sq.sum(axis=0)[None, :] - sq1
    var1 = np.divide(sq1 - n1 * mean1 * mean1, n1 - 1, out=np.full_like(sum1, np.nan), where=n1 > 1)
    var0 = np.divide(sq0 - n0 * mean0 * mean0, n0 - 1, out=np.full_like(sum0, np.nan), where=n0 > 1)
    pv = np.divide(
        (n1 - 1) * var1 + (n0 - 1) * var0,
        n1 + n0 - 2,
        out=np.full_like(sum1, np.nan),
        where=(n1 > 1) & (n0 > 1),
    )
    sd = np.sqrt(np.maximum(pv, 0.0))
    perm_d = np.divide(
        mean1 - mean0,
        sd,
        out=np.full_like(mean1, np.nan),
        where=(n1 >= MIN_DONORS_PER_GROUP) & (n0 >= MIN_DONORS_PER_GROUP) & (sd > 0),
    )
    return observed_d, perm_d, n0_obs, n1_obs


def permutation_selection(
    score_map: Dict[str, float],
    donor_data: dict,
    *,
    n_permutations: int = PERMUTATION_N,
    alpha: float = PERMUTATION_ALPHA,
    effect_size: float = PERMUTATION_EFFECT_SIZE,
    random_state: int = 42,
    cssi_mode: bool = False,
):
    """Return selected genes plus per-gene d/p/q audit data."""
    items = _ranked_items(score_map)
    genes = [str(g) for g in donor_data["genes"]]
    gene_to_col = {g: j for j, g in enumerate(genes)}
    labels = np.asarray(donor_data["labels"], dtype=int)
    D = len(labels)
    n_pd = int(np.sum(labels == 1))
    n_ctrl = D - n_pd
    if n_pd < MIN_DONORS_PER_GROUP or n_ctrl < MIN_DONORS_PER_GROUP:
        raise ValueError(
            f"Donor permutation requires >= {MIN_DONORS_PER_GROUP} donors per condition; "
            f"got PD={n_pd}, control={n_ctrl}."
        )

    rng = np.random.default_rng(random_state)
    perm_labels = _permutation_labels(D, n_pd, n_permutations, rng)
    G = len(genes)
    observed_d = np.full(G, np.nan)
    observed_n0 = np.zeros(G, dtype=int)
    observed_n1 = np.zeros(G, dtype=int)
    null_stat = np.full((n_permutations, G), np.nan)
    observed_stat = np.full(G, np.nan)

    if not cssi_mode:
        d, perm_d, n0, n1 = _permute_matrix_d(
            np.asarray(donor_data["scores"], dtype=float), labels, perm_labels
        )
        observed_d[:] = d
        observed_n0[:] = n0
        observed_n1[:] = n1
        observed_stat[:] = np.abs(d)
        null_stat[:] = np.abs(perm_d)
        test_name = "two-sided donor-label Cohen's d"
    else:
        # Max-|d| over strata. The same max is applied to each permutation.
        for _, matrix in sorted(donor_data["strata"].items(), key=lambda kv: kv[0]):
            d, perm_d, n0, n1 = _permute_matrix_d(
                np.asarray(matrix, dtype=float), labels, perm_labels
            )
            abs_d = np.abs(d)
            replace = np.isfinite(abs_d) & (
                ~np.isfinite(observed_stat) | (abs_d > observed_stat)
            )
            observed_stat[replace] = abs_d[replace]
            observed_d[replace] = d[replace]
            observed_n0[replace] = n0[replace]
            observed_n1[replace] = n1[replace]
            null_stat = np.fmax(null_stat, np.abs(perm_d))
        test_name = "two-sided donor-label Cohen's d, max-|d| over CSSI strata"

    p = np.full(G, np.nan)
    for j in range(G):
        if not np.isfinite(observed_stat[j]):
            continue
        null_j = null_stat[:, j]
        null_j = null_j[np.isfinite(null_j)]
        if null_j.size:
            threshold = observed_stat[j] * (1.0 - TIE_RTOL)
            p[j] = (1 + np.sum(null_j >= threshold)) / (null_j.size + 1)
    q = np.full(G, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        q[finite] = _bh_adjust(p[finite])

    audit = []
    selected = []
    for gene, score in items:
        j = gene_to_col.get(gene)
        if j is None or not np.isfinite(q[j]) or not np.isfinite(observed_d[j]):
            continue
        
        q_pass = bool(q[j] < alpha)
        effect_pass = bool(abs(observed_d[j]) >= effect_size)
        row = {
            "gene": gene,
            "score": float(score),
            "cohen_d": float(observed_d[j]),
            "p_value": float(p[j]),
            "q_value": float(q[j]),
            "n_donors_control": int(observed_n0[j]),
            "n_donors_pd": int(observed_n1[j]),
            "effect_threshold": float(effect_size),
            "q_pass": q_pass,
            "effect_pass": effect_pass,
            "selected": q_pass and effect_pass,
        }
        audit.append(row)
        if row["selected"]:
            selected.append((gene, float(score), row))

    selected.sort(key=lambda x: (-x[1], x[0]))
    selected_simple = [(g, s) for g, s, _ in selected]
    meta = {
        "n_permutations": int(n_permutations),
        "alpha": float(alpha),
        "effect_size_threshold": float(effect_size),
        "selection_rule": "q < alpha AND abs(Cohen's d) >= effect_size",
        "two_sided": True,
        "multiple_testing": "Benjamini-Hochberg",
        "n_donors": D,
        "n_donors_control": n_ctrl,
        "n_donors_pd": n_pd,
        "min_donors_per_group": MIN_DONORS_PER_GROUP,
        "test": test_name,
        "n_selected": len(selected_simple),
    }
    return selected_simple, meta, audit


def select_all(score_map: Dict[str, float]):
    items = _ranked_items(score_map)
    if not items:
        raise ValueError("Cannot select genes from an empty score map.")
    return {
        "top30": (_select_top_k(items, TOP_N), {"k": min(TOP_N, len(items))}),
        "z-score": _zscore(items),
        "elbow": _elbow(items),
    }


def write_selection_csv(selected, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "gene", "score"])
        for rank, (gene, score) in enumerate(selected, 1):
            writer.writerow([rank, gene, score])


def write_permutation_csv(audit_rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(audit_rows, key=lambda r: (-r["score"], r["gene"]))
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank", "gene", "score", "cohen_d", "p_value", "q_value",
            "n_donors_control", "n_donors_pd", "effect_threshold",
            "q_pass", "effect_pass", "selected",
        ])
        for rank, row in enumerate(selected, 1):
            writer.writerow([
                rank, row["gene"], row["score"], row["cohen_d"], row["p_value"],
                row["q_value"], row["n_donors_control"], row["n_donors_pd"],
                row["effect_threshold"], row.get("q_pass", False),
                row.get("effect_pass", False), row.get("selected", False),
            ])


def write_selection_family(
    score_map: Dict[str, float],
    out_dir: Path,
    *,
    donor_data: Optional[dict] = None,
    cssi_mode: bool = False,
    random_state: int = 42,
    permutation_n: int = PERMUTATION_N,
    permutation_alpha: float = PERMUTATION_ALPHA,
    permutation_effect_size: float = PERMUTATION_EFFECT_SIZE,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    selections = select_all(score_map)
    summary = {}
    for method, (selected, metadata) in selections.items():
        path = out_dir / f"{method}.csv"
        write_selection_csv(selected, path)
        summary[method] = {**metadata, "n_selected": len(selected), "output": str(path)}

    # Permutation is opt-in: without donor data only the fixed selectors run
    if donor_data is None:
        with open(out_dir / "selection_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    _, perm_meta, audit_all = permutation_selection(
        score_map,
        donor_data,
        n_permutations=permutation_n,
        alpha=permutation_alpha,
        effect_size=permutation_effect_size,
        random_state=random_state,
        cssi_mode=cssi_mode,
    )
    # `permutation.csv` remains the selected gene set. A separate audit file
    # contains every gene so q-values and effect sizes can be inspected even
    # when the final selection is empty.
    selected_audit = [row for row in audit_all if row["selected"]]
    perm_path = out_dir / "permutation.csv"
    write_permutation_csv(selected_audit, perm_path)

    audit_path = out_dir / "permutation_audit.csv"
    write_permutation_csv(audit_all, audit_path)

    n_q_pass = sum(row["q_pass"] for row in audit_all)
    n_effect_pass = sum(row["effect_pass"] for row in audit_all)
    summary["permutation"] = {
        **perm_meta,
        "n_audit_genes": len(audit_all),
        "n_q_pass": int(n_q_pass),
        "n_effect_pass": int(n_effect_pass),
        "n_selected": len(selected_audit),
        "output": str(perm_path),
        "audit_output": str(audit_path),
    }
    with open(out_dir / "selection_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary
