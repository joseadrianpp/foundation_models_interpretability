"""Comparison plots between interpretability methods, per model and gene-set selection.

Every plot is written under results/dataset_*/plots/, one figure per model
(`<name>_<model>.png`); top-30 plots have no selection suffix and other
selections get a `_<selection>` suffix (see gene_sets.plot_path). Each model has
nine methods: three rankings (pooled / CSSI / OLS) of three scores (zero-shot
attention, LoRA attention, LoRA IG). The color is the ranking and the texture
the score (solid / //// / xxxx), never the position, so a method looks the
same in every figure, model and selection.

Every public plot_* function loops over the models and skips a model whose
inputs are missing, so callers do not need to know which models were run.
"""

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gene_sets import (ALL_SCORE_PATHS, CELL_TYPES, METHOD_META, METHODS, MODELS, SELECTIONS,
                        SUMMARY_PATHS, load_genes, methods_for, output_dir, plot_path)

MODEL_LABELS = {"scgpt": "scGPT", "geneformer": "Geneformer", "sccello": "scCello"}


# Ranking (pooled / CSSI / OLS) and family (form and score: zero-shot attention,
# Step 2.5 attention, zero-shot + MLP attention or IG, LoRA attention or IG) of a method
def _ranking(method: str) -> str:
    ranking = METHOD_META[method]["ranking"]
    return ranking if ranking in ("cssi", "ols") else "pooled"


def _family(method: str) -> tuple:
    meta = METHOD_META[method]
    return meta["model_form"], meta["score"]


# More methods per model than the categorical palette holds, so the encoding
# is composite: the first three slots of the fixed categorical order (blue,
# orange, aqua; CVD safe for every pair, validated on the plot surface) for
# the ranking, and a texture for the form and score (solid: zero-shot attention)
RANKING_COLORS = {"pooled": "#2a78d6", "cssi": "#eb6834", "ols": "#1baf7a"}
RANKING_SHORT = {"pooled": "", "cssi": "CSSI", "ols": "OLS"}
FORM_SHORT = {"zero-shot": "zero-shot", "step2_5": "Step 2.5", "mlp": "MLP", "lora": "LoRA"}
SCORE_SHORT = {"attention": "attn.", "ig": "IG", "igsigned": "IG±", "rollout": "roll.", "rolloutsigned": "roll.±",
               "gradrollout": "g.roll.", "gradrolloutsigned": "g.roll.±"}
ZERO_SHOT_ATTENTION = ("zero-shot", "attention")
_HATCHES = ["////", "xxxx", "....", "\\\\", "++++", "oo", "**", "||||", "----", "O.", "x.", "/.", "+.", "\\."]
_FAMILIES = [f for f in dict.fromkeys(_family(m) for m in METHODS) if f != ZERO_SHOT_ATTENTION]
FAMILY_HATCH = {ZERO_SHOT_ATTENTION: None, **{f: _HATCHES[i % len(_HATCHES)] for i, f in enumerate(_FAMILIES)}}
FAMILY_SHORT = {f: "" if f == ZERO_SHOT_ATTENTION else f"{FORM_SHORT[f[0]]} {SCORE_SHORT[f[1]]}"
                for f in FAMILY_HATCH}
METHOD_COLORS = {m: RANKING_COLORS[_ranking(m)] for m in METHODS}
METHOD_HATCH = {m: FAMILY_HATCH[_family(m)] for m in METHODS}
# Labels inside a figure: the model is already in the title
METHOD_LABELS = {m: METHOD_META[m]["label"] for m in METHODS}
METHOD_SHORT = {m: (" ".join(x for x in (FAMILY_SHORT[_family(m)], RANKING_SHORT[_ranking(m)]) if x)
                    or "Pooled") for m in METHODS}

CELL_TYPE_LABELS = {"astrocytes": "Astrocytes", "da_neurons": "DA neurons",
                    "microglia": "Microglia", "oligodendrocytes": "Oligodendrocytes"}
SELECTION_LABELS = {"top30": "static top-30", "z-score": "z-score (mean + 2.5 sd)",
                    "elbow": "elbow (Kneedle)"}
SELECTION_HATCH = {"top30": None, "z-score": "....", "elbow": "////"}
# Parkinson's references of gwas_similarity.py / gene_matrix.py
PD_REFERENCES = {"parkinson": "PD GWAS", "panelapp": "PD PanelApp",
                 "parkinsonsuk_ucl": "PD ParkinsonsUK-UCL GO"}
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
})


def _stars_for(p: float) -> str:
    for threshold, label in SIG_LEVELS:
        if p < threshold:
            return label
    return ""


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _method_handle(method: str):
    return plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[method], hatch=METHOD_HATCH[method],
                         edgecolor=SURFACE)


def _method_legend(fig, methods, ncol=None) -> None:
    # One row per score (three rankings each)
    handles = [_method_handle(m) for m in methods]
    fig.legend(handles, [METHOD_LABELS[m] for m in methods], loc="upper center",
               ncol=ncol or min(3, len(methods)), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _bar_offsets(n_methods: int):
    width = 0.8 / n_methods
    return width, [(i - (n_methods - 1) / 2) * width for i in range(n_methods)]


class _NoData(Exception):
    """A model has nothing to plot (its tasks were not run)."""


# Run one per-model plot for every model, skipping models without inputs
def _for_models(name: str, plot_one, *args) -> list:
    paths = []
    for model in MODELS:
        try:
            paths.append(plot_one(model, *args))
        except (_NoData, FileNotFoundError) as exc:
            print(f"  skipped {name} [{model}]: {exc}")
    return paths


# ---------------------------------------------------------------------------
# Enrichment summary
# ---------------------------------------------------------------------------
def _draw_enrichment_panel(ax, summary: pd.DataFrame, tests: pd.DataFrame, methods: list,
                           metric: str, metric_key: str, title: str) -> None:
    x = np.arange(len(CELL_TYPES))
    width, offsets = _bar_offsets(len(methods))
    tops = {}
    for method, off in zip(methods, offsets):
        heights = []
        for cell_type in CELL_TYPES:
            row = summary[(summary["cell_type"] == cell_type) & (summary["method"] == method)]
            val = row[metric].iloc[0] if len(row) else 0.0
            val = 0.0 if pd.isna(val) else float(val)
            heights.append(val)
            tops[(cell_type, method)] = val
        ax.bar(x + off, heights, width=width * 0.92, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], hatch=METHOD_HATCH[method], edgecolor=SURFACE, lw=0.5)

    # n_significant_terms has no pairwise test (only IC/intersection_size/functional abundance do).
    # Brackets mark method pairs with BH-adjusted p < 0.05, stacked above the bars.
    ymax = max(tops.values(), default=0.0)
    if metric_key is not None and ymax > 0:
        step = ymax * 0.1
        top_bracket = ymax
        pos = {m: off for m, off in zip(methods, offsets)}
        for i, cell_type in enumerate(CELL_TYPES):
            sig = tests[(tests["cell_type"] == cell_type) & (tests["metric"] == metric_key)
                        & (tests["p_adj_bh"] < 0.05)]
            level = max(tops[(cell_type, m)] for m in methods) + step
            for r in sig.itertuples():
                if r.method_a not in pos or r.method_b not in pos:
                    continue
                x1, x2 = x[i] + pos[r.method_a], x[i] + pos[r.method_b]
                ax.plot([x1, x1, x2, x2], [level - step * 0.3, level, level, level - step * 0.3],
                        color=INK, lw=0.8)
                ax.text((x1 + x2) / 2, level, _stars_for(r.p_adj_bh), ha="center", va="bottom", fontsize=8)
                top_bracket = max(top_bracket, level + step)
                level += step * 1.2
        ax.set_ylim(top=max(ymax * 1.15, top_bracket * 1.05))

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_TYPE_LABELS[c] for c in CELL_TYPES], rotation=20, ha="right")
    ax.set_title(title, fontsize=11, fontweight="bold")
    _style(ax)


def _plot_enrichment_summary(model: str, results_dir: Path, selection: str) -> Path:
    summary_dir = output_dir(results_dir, "enrichment_summary", selection)
    summary = pd.read_csv(summary_dir / "summary_table.csv")
    tests = pd.read_csv(summary_dir / "pairwise_tests.csv")
    tests = tests[tests["model"] == model]
    methods = [m for m in methods_for(selection, model)
               if m in set(summary.loc[summary["n_query_genes"].notna(), "method"])]
    if not methods:
        raise _NoData("no enriched gene sets")

    panels = [
        ("n_significant_terms", None, "Number of significant\nGO/KEGG/REAC terms"),
        ("mean_information_content", "information_content", "Mean information content\n(significant terms)"),
        ("mean_intersection_size", "intersection_size", "Mean intersection size\n(genes per term)"),
        ("mean_functional_abundance", "functional_abundance", "Mean functional abundance\n(fraction of the term covered)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4.8))
    for ax, (metric, metric_key, title) in zip(axes, panels):
        _draw_enrichment_panel(ax, summary, tests, methods, metric, metric_key, title)
    fig.suptitle(f"{MODEL_LABELS[model]} - g:Profiler enrichment - {SELECTION_LABELS[selection]} gene sets "
                 "(brackets: Mann-Whitney, BH-adjusted p)", fontsize=12)
    _method_legend(fig, methods)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, plot_path(results_dir, f"enrichment_summary_{model}", selection))


def plot_enrichment_summary(results_dir: Path, selection: str = "top30") -> list:
    return _for_models(f"enrichment_summary/{selection}", _plot_enrichment_summary, results_dir, selection)


# ---------------------------------------------------------------------------
# PD similarity (gene-level best match, from gene_matrix), one figure per PD reference
# ---------------------------------------------------------------------------
ONTOLOGIES = ["BP", "CC", "MF"]
DISTANCE = "SimRel"
DISEASE = "parkinson"

# Per-gene closest match (row max) to one PD reference, tagged by method;
# read straight from gene_matrix.py's matrix, not the pre-aggregated summary.
# Unannotated genes have all-NaN rows and no best match, so they are left out
def _bestmatch_per_gene(results_dir: Path, selection: str, reference: str, cell_type: str,
                        ontology: str) -> pd.DataFrame:
    path = (output_dir(results_dir, "gene_matrix", selection) / "matrices" / cell_type
            / f"{reference}__{DISTANCE}__{ontology}.csv")
    mat = pd.read_csv(path, index_col=[0, 1])
    return mat.max(axis=1).dropna().rename("bestmatch").reset_index()

def _draw_gwas_panel(ax, results_dir: Path, selection: str, reference: str, methods: list,
                     ontology: str) -> None:
    data, positions, colors, hatches = [], [], [], []
    tick_positions, tick_labels = [], []
    pos = 0.0
    for cell_type in CELL_TYPES:
        df = _bestmatch_per_gene(results_dir, selection, reference, cell_type, ontology)
        group_start = pos
        for method in methods:
            vals = df.loc[df["method"] == method, "bestmatch"].to_numpy()
            if len(vals) > 0:
                data.append(vals)
                positions.append(pos)
                colors.append(METHOD_COLORS[method])
                hatches.append(METHOD_HATCH[method])
            pos += 1.0
        tick_positions.append((group_start + pos - 1.0) / 2)
        tick_labels.append(CELL_TYPE_LABELS[cell_type])
        pos += 0.8  # gap between cell types

    if not data:
        raise _NoData("no gene-matrix rows for this model")
    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
    for patch, color, hatch in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(color)
        patch.set_hatch(hatch)
        patch.set_edgecolor(SURFACE)
    for median in bp["medians"]:
        median.set_color(INK)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_title(f"GO:{ontology}", fontsize=11, fontweight="bold")
    _style(ax)

def _plot_gwas_similarity(model: str, results_dir: Path, selection: str, reference: str) -> Path:
    methods = methods_for(selection, model)
    fig, axes = plt.subplots(1, len(ONTOLOGIES), figsize=(15, 4.8), sharey=True)
    try:
        for ax, ontology in zip(axes, ONTOLOGIES):
            _draw_gwas_panel(ax, results_dir, selection, reference, methods, ontology)
    except (_NoData, FileNotFoundError):
        plt.close(fig)
        raise
    label = PD_REFERENCES[reference]
    axes[0].set_ylabel(f"Best-match similarity to {label} genes\n({DISTANCE}, per gene)", fontsize=10)
    fig.suptitle(f"{MODEL_LABELS[model]} - closest {label} gene per selected gene - "
                 f"{SELECTION_LABELS[selection]}", fontsize=12)
    _method_legend(fig, methods)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, plot_path(results_dir, f"gwas_similarity_{reference}_{model}", selection))


def plot_gwas_similarity(results_dir: Path, selection: str = "top30") -> list:
    paths = []
    for reference in PD_REFERENCES:
        paths += _for_models(f"gwas_similarity/{reference}/{selection}", _plot_gwas_similarity,
                             results_dir, selection, reference)
    return paths


# ---------------------------------------------------------------------------
# Disease set-level similarity (BMA, from gwas_similarity): PD GWAS bars, the
# two PD panels and the control diseases as markers
# ---------------------------------------------------------------------------
REFERENCE_MARKERS = [  # (reference, marker, filled, legend label)
    ("panelapp", "^", True, "vs PD PanelApp"),
    ("parkinsonsuk_ucl", "s", True, "vs PD ParkinsonsUK-UCL GO"),
    ("alzheimer", "o", False, "vs Alzheimer (control)"),
    ("cad", "x", True, "vs CAD (control)"),
    ("kidney", "D", False, "vs CKD (control)"),
]


def _plot_gwas_set_similarity(model: str, results_dir: Path, selection: str) -> Path:
    df = pd.read_csv(output_dir(results_dir, "gwas_similarity", selection) / "gene_level_similarity_all.csv")
    df = df[df["distance"] == DISTANCE]
    methods = [m for m in methods_for(selection, model) if m in set(df["method"])]
    if not methods:
        raise _NoData("no similarity rows for this model")
    x = np.arange(len(CELL_TYPES))
    width, offsets = _bar_offsets(len(methods))

    fig, axes = plt.subplots(1, len(ONTOLOGIES), figsize=(15, 4.8), sharey=True)
    for ax, ontology in zip(axes, ONTOLOGIES):
        sub = df[df["go_source"] == f"GO:{ontology}"].set_index(["reference", "method", "cell_type"])["similarity"]
        for method, off in zip(methods, offsets):
            pd_vals = [sub.get(("parkinson", method, c), np.nan) for c in CELL_TYPES]
            ax.bar(x + off, pd_vals, width=width * 0.92, color=METHOD_COLORS[method],
                   hatch=METHOD_HATCH[method], edgecolor=SURFACE, lw=0.5)
            # Other PD references and control diseases: the PD GWAS bar only
            # means something relative to the controls
            for ref, marker, filled, _ in REFERENCE_MARKERS:
                vals = [sub.get((ref, method, c), np.nan) for c in CELL_TYPES]
                ax.scatter(x + off, vals, marker=marker, s=22, zorder=3, linewidths=1,
                           color=INK, facecolors=INK if filled else "none")
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_TYPE_LABELS[c] for c in CELL_TYPES], rotation=20, ha="right")
        ax.set_title(f"GO:{ontology}", fontsize=11, fontweight="bold")
        _style(ax)
    axes[0].set_ylabel(f"Gene-set similarity to reference genes\n({DISTANCE}, BMA)", fontsize=10)

    handles = [_method_handle(m) for m in methods]
    labels = [f"{METHOD_LABELS[m]} vs PD GWAS (bar)" for m in methods]
    for _, marker, filled, label in REFERENCE_MARKERS:
        handles.append(plt.Line2D([], [], ls="", marker=marker, color=INK,
                                  mfc=INK if filled else "none", mec=INK))
        labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"{MODEL_LABELS[model]} - disease gene-set similarity - {SELECTION_LABELS[selection]}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, plot_path(results_dir, f"gwas_set_similarity_{model}", selection))


def plot_gwas_set_similarity(results_dir: Path, selection: str = "top30") -> list:
    return _for_models(f"gwas_set_similarity/{selection}", _plot_gwas_set_similarity, results_dir, selection)


# ---------------------------------------------------------------------------
# Gene-set sizes and overlap between methods
# ---------------------------------------------------------------------------
def _jaccard(a: list, b: list) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def _plot_gene_set_overlap(model: str, results_dir: Path, selection: str) -> Path:
    methods = methods_for(selection, model)
    sets_by_ct = {ct: {m: load_genes(results_dir, m, selection, ct) for m in methods} for ct in CELL_TYPES}
    if not any(s for sets in sets_by_ct.values() for s in sets.values()):
        raise _NoData("no gene lists")
    fig, axes = plt.subplots(1, len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 4.2))
    for ax, cell_type in zip(axes, CELL_TYPES):
        sets = sets_by_ct[cell_type]
        n = len(methods)
        mat = np.full((n, n), np.nan)
        for i, j in itertools.product(range(n), range(n)):
            if sets[methods[i]] and sets[methods[j]]:
                mat[i, j] = _jaccard(sets[methods[i]], sets[methods[j]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        for i, j in itertools.product(range(n), range(n)):
            if np.isfinite(mat[i, j]):
                shared = len(set(sets[methods[i]]) & set(sets[methods[j]]))
                ax.text(j, i, f"{mat[i, j]:.2f}\n({shared})", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > 0.55 else INK)
        labels = [f"{METHOD_SHORT[m]}\nn={len(sets[m]) if sets[m] else 0}" for m in methods]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(CELL_TYPE_LABELS[cell_type], fontsize=11, fontweight="bold")
    fig.suptitle(f"{MODEL_LABELS[model]} - gene-set overlap between methods - {SELECTION_LABELS[selection]} "
                 "(Jaccard index, shared genes in brackets)", fontsize=12)
    fig.tight_layout()
    return _save(fig, plot_path(results_dir, f"gene_set_overlap_{model}", selection))


def plot_gene_set_overlap(results_dir: Path, selection: str = "top30") -> list:
    return _for_models(f"gene_set_overlap/{selection}", _plot_gene_set_overlap, results_dir, selection)


def _summary(results_dir: Path, method: str, cell_type: str):
    path = results_dir / SUMMARY_PATHS[method].format(ct=cell_type)
    return json.loads(path.read_text()) if path.exists() else None


def _plot_gene_set_sizes(model: str, results_dir: Path) -> Path:
    """Number of genes each method selects per selection rule."""
    methods = [m for m in SUMMARY_PATHS if METHOD_META[m]["model"] == model]
    if not any(_summary(results_dir, m, ct) for m in methods for ct in CELL_TYPES):
        raise _NoData("no selection summaries")
    rules = SELECTIONS
    fig, axes = plt.subplots(1, len(rules), figsize=(4 * len(rules), 4.2), sharey=True)
    x = np.arange(len(CELL_TYPES))
    width, offsets = _bar_offsets(len(methods))
    for ax, rule in zip(axes, rules):
        for method, off in zip(methods, offsets):
            vals = []
            for cell_type in CELL_TYPES:
                summary = _summary(results_dir, method, cell_type)
                vals.append(np.nan if summary is None else summary.get(rule, {}).get("n_selected", np.nan))
            bars = ax.bar(x + off, vals, width=width * 0.92, color=METHOD_COLORS[method],
                          hatch=METHOD_HATCH[method], edgecolor=SURFACE, lw=0.5)
            # Vertical: nine bars per cell type leave no room for horizontal labels
            ax.bar_label(bars, labels=["" if pd.isna(v) else int(v) for v in vals],
                         fontsize=6, color=INK_2, padding=1, rotation=90)
        ax.margins(y=0.12)
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_TYPE_LABELS[c] for c in CELL_TYPES], rotation=20, ha="right")
        ax.set_title(SELECTION_LABELS[rule], fontsize=11, fontweight="bold")
        _style(ax)
    axes[0].set_ylabel("Genes selected")
    fig.suptitle(f"{MODEL_LABELS[model]} - gene-set size per selection rule", fontsize=12)
    _method_legend(fig, methods)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, results_dir / "plots" / f"gene_set_sizes_{model}.png")


def plot_gene_set_sizes(results_dir: Path) -> list:
    return _for_models("gene_set_sizes", _plot_gene_set_sizes, results_dir)


# ---------------------------------------------------------------------------
# Dynamic cut-offs on each method's ranked score curve
# ---------------------------------------------------------------------------
CUTOFF_STYLES = {"top30": ":", "z-score": "--", "elbow": "-"}
MAX_RANK_SHOWN = 200


def _all_scores(results_dir: Path, method: str, cell_type: str):
    path = results_dir / ALL_SCORE_PATHS[method].format(ct=cell_type)
    if not path.exists():
        return None
    return np.sort(pd.read_csv(path)["score"].to_numpy(dtype=float))[::-1]


def _plot_dynamic_selection(model: str, results_dir: Path) -> Path:
    methods = [m for m in ALL_SCORE_PATHS if METHOD_META[m]["model"] == model]
    if not any(_all_scores(results_dir, m, ct) is not None for m in methods for ct in CELL_TYPES):
        raise _NoData("no all-gene score files")
    fig, axes = plt.subplots(len(methods), len(CELL_TYPES), squeeze=False,
                             figsize=(4 * len(CELL_TYPES), 3.2 * len(methods)))
    for r, method in enumerate(methods):
        for c, cell_type in enumerate(CELL_TYPES):
            ax = axes[r, c]
            scores = _all_scores(results_dir, method, cell_type)
            summary = _summary(results_dir, method, cell_type)
            if scores is None or summary is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                ax.set_axis_off()
                continue
            shown = scores[:MAX_RANK_SHOWN]
            ax.plot(np.arange(1, len(shown) + 1), shown, color=METHOD_COLORS[method], lw=2)
            for rule, ls in CUTOFF_STYLES.items():
                k = summary.get(rule, {}).get("n_selected")
                if k:
                    ax.axvline(k + 0.5, color=INK_2, ls=ls, lw=1, label=f"{rule} (n={k})")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
            if r == 0:
                ax.set_title(CELL_TYPE_LABELS[cell_type], fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{METHOD_SHORT[method]} score")
            if r == len(methods) - 1:
                ax.set_xlabel(f"Gene rank (first {MAX_RANK_SHOWN})")
            _style(ax)
    fig.suptitle(f"{MODEL_LABELS[model]} - ranked score curves and dynamic cut-offs per method",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, results_dir / "plots" / f"dynamic_selection_cutoffs_{model}.png")


def plot_dynamic_selection(results_dir: Path) -> list:
    return _for_models("dynamic_selection_cutoffs", _plot_dynamic_selection, results_dir)


# ---------------------------------------------------------------------------
# Method 2.3 diagnostics: what OLS removes from the attention score
# ---------------------------------------------------------------------------
OLS_COVARIATES = {"scgpt": "expression mean/variance/dropout",
                  "geneformer": "rank pseudo-expression mean/variance/dropout",
                  "sccello": "rank pseudo-expression mean/variance/dropout"}


def _plot_ols_residualization(model: str, results_dir: Path) -> Path:
    bases = {ct: results_dir / f"attention_residualized_ols_{model}" / ct for ct in CELL_TYPES}
    if not any((b / "residualized_genes.csv").exists() for b in bases.values()):
        raise _NoData("no OLS outputs")
    fig, axes = plt.subplots(2, len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 7.5))
    for c, cell_type in enumerate(CELL_TYPES):
        base = bases[cell_type]
        path = base / "residualized_genes.csv"
        if not path.exists():
            for ax in axes[:, c]:
                ax.set_axis_off()
            continue
        df = pd.read_csv(path)
        meta_path = base / "ols_summary.json"
        r2 = json.loads(meta_path.read_text()).get("r_squared", np.nan) if meta_path.exists() else np.nan
        top_raw = set(df.nlargest(30, "attention_score")["gene"])
        top_res = set(df.nlargest(30, "residual_attention")["gene"])
        in_both = df["gene"].isin(top_raw & top_res)
        only_res = df["gene"].isin(top_res - top_raw)
        only_raw = df["gene"].isin(top_raw - top_res)
        rest = ~(in_both | only_res | only_raw)

        for r, ycol in enumerate(["predicted_attention", "residual_attention"]):
            ax = axes[r, c]
            ax.scatter(df.loc[rest, "attention_score"], df.loc[rest, ycol], s=6, color="#c3c2b7", lw=0)
            ax.scatter(df.loc[only_raw, "attention_score"], df.loc[only_raw, ycol], s=18,
                       color=METHOD_COLORS[f"attention_zero_shot_{model}"], edgecolors=SURFACE, lw=0.6,
                       label="top-30 raw only")
            ax.scatter(df.loc[only_res, "attention_score"], df.loc[only_res, ycol], s=18,
                       color=METHOD_COLORS[f"attention_residualized_ols_{model}"], edgecolors=SURFACE, lw=0.6,
                       label="top-30 residual only")
            ax.scatter(df.loc[in_both, "attention_score"], df.loc[in_both, ycol], s=18,
                       color=INK, edgecolors=SURFACE, lw=0.6, label="top-30 in both")
            if r == 1:
                lims = [df["attention_score"].min(), df["attention_score"].max()]
                ax.plot(lims, lims, color=MUTED, lw=0.8, ls="--")
            ax.axhline(0, color=GRID, lw=0.8)
            if r == 0:
                ax.set_title(f"{CELL_TYPE_LABELS[cell_type]} (R$^2$={r2:.3f}, "
                             f"top-30 overlap {len(top_raw & top_res)}/30)", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel("predicted_attention (OLS fit)" if r == 0 else "residual_attention")
            if r == 1:
                ax.set_xlabel("attention_score (PD - control)")
            _style(ax)
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes[1] if ax.get_legend_handles_labels()[0]),
                           ([], []))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{MODEL_LABELS[model]} - method 2.3: attention explained by {OLS_COVARIATES[model]} (top) "
                 "vs residual kept for ranking (bottom; dashed y = x)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, results_dir / "plots" / f"ols_residualization_{model}.png")


def plot_ols_residualization(results_dir: Path) -> list:
    return _for_models("ols_residualization", _plot_ols_residualization, results_dir)


# ---------------------------------------------------------------------------
# Selections compared per method: set size, enrichment and PD similarity
# ---------------------------------------------------------------------------
GWAS_ONTOLOGY = "BP"


def _selection_metrics(results_dir: Path, selection: str) -> pd.DataFrame:
    """One row per (method, cell_type): query size, enrichment terms, PD similarity."""
    manifest = pd.read_csv(output_dir(results_dir, "enrichment", selection) / "manifest.csv")
    gwas = pd.read_csv(output_dir(results_dir, "gwas_similarity", selection) / "gene_level_similarity_all.csv")
    gwas = gwas[(gwas["reference"] == DISEASE) & (gwas["distance"] == DISTANCE)
                & (gwas["go_source"] == f"GO:{GWAS_ONTOLOGY}")]
    df = manifest.merge(gwas[["method", "cell_type", "similarity"]], on=["method", "cell_type"], how="outer")
    df["selection"] = selection
    return df


def _plot_selection_comparison(model: str, results_dir: Path, df: pd.DataFrame, selections) -> Path:
    methods = [m for m in METHODS if METHOD_META[m]["model"] == model and m in set(df["method"])]
    if not methods:
        raise _NoData("no enrichment/similarity rows for this model")
    rows = [("n_query_genes", "Genes in set"),
            ("n_significant_terms", "Significant\nGO/KEGG/REAC terms"),
            ("similarity", f"PD GWAS similarity\n({DISTANCE}, BMA, GO:{GWAS_ONTOLOGY})")]

    fig, axes = plt.subplots(len(rows), len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 3.1 * len(rows)),
                             sharey="row")
    x = np.arange(len(methods))
    width, offsets = _bar_offsets(len(selections))
    for r, (metric, ylabel) in enumerate(rows):
        for c, cell_type in enumerate(CELL_TYPES):
            ax = axes[r, c]
            sub = df[df["cell_type"] == cell_type].set_index(["method", "selection"])[metric]
            for selection, off in zip(selections, offsets):
                vals = [sub.get((m, selection), np.nan) for m in methods]
                bars = ax.bar(x + off, vals, width=width * 0.9, color=[METHOD_COLORS[m] for m in methods],
                              hatch=SELECTION_HATCH[selection], edgecolor=SURFACE, lw=0.8)
                if metric != "similarity":
                    ax.bar_label(bars, labels=["" if pd.isna(v) else int(v) for v in vals],
                                 fontsize=7, color=INK_2, padding=1)
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_SHORT[m] for m in methods], fontsize=7, rotation=45, ha="right")
            if r == 0:
                ax.set_title(CELL_TYPE_LABELS[cell_type], fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            _style(ax)

    # Bars are colored by ranking; the method of each bar is its tick label
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in RANKING_COLORS.values()]
    labels = ["pooled score", "CSSI-max", "OLS residual"]
    for selection in selections:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK_2,
                                     hatch=SELECTION_HATCH[selection]))
        labels.append(SELECTION_LABELS[selection])
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{MODEL_LABELS[model]} - gene sets per method and selection", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, results_dir / "plots" / f"selection_comparison_{model}.png")


def plot_selection_comparison(results_dir: Path, selections=SELECTIONS) -> list:
    df = pd.concat([_selection_metrics(results_dir, s) for s in selections], ignore_index=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "plots" / "selection_comparison.csv", index=False)
    return _for_models("selection_comparison", _plot_selection_comparison, results_dir, df, selections)


def task_plots(results_dir: Path, selections=SELECTIONS) -> None:
    """Rebuild every comparison plot from the CSVs already on disk (no API calls)."""
    steps = [("gene_set_sizes", lambda: plot_gene_set_sizes(results_dir)),
             ("dynamic_selection_cutoffs", lambda: plot_dynamic_selection(results_dir)),
             ("ols_residualization", lambda: plot_ols_residualization(results_dir)),
             ("selection_comparison", lambda: plot_selection_comparison(results_dir, selections))]
    for selection in selections:
        steps += [
            (f"gene_set_overlap/{selection}", lambda s=selection: plot_gene_set_overlap(results_dir, s)),
            (f"enrichment_summary/{selection}", lambda s=selection: plot_enrichment_summary(results_dir, s)),
            (f"gwas_set_similarity/{selection}", lambda s=selection: plot_gwas_set_similarity(results_dir, s)),
            (f"gwas_similarity/{selection}", lambda s=selection: plot_gwas_similarity(results_dir, s)),
        ]
    for name, fn in steps:
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  skipped {name}: {exc}")


def plot_gene_set_overview(results_dir: Path, selections=SELECTIONS) -> None:
    plot_gene_set_sizes(results_dir)
    plot_dynamic_selection(results_dir)
    plot_ols_residualization(results_dir)
    for selection in selections:
        plot_gene_set_overlap(results_dir, selection)
