"""Comparison plots between interpretability methods, per gene-set selection.

Every plot is written under results/plots/; top-30 plots keep their original
names and other selections get a `_<selection>` suffix (see gene_sets.plot_path).
Colors follow the method (never its position), so the same method has the same
color in every figure and selection.
"""

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gene_sets import CELL_TYPES, METHODS as METHODS_ORDER, SELECTIONS, load_genes, methods_for, output_dir, plot_path

METHOD_LABELS = {
    "attention_zero_shot_scgpt": "scGPT zero-shot attention (pooled)",
    "attention_cssi_scgpt": "scGPT CSSI-max attention (2.1)",
    "attention_residualized_ols_scgpt": "scGPT OLS-residual attention (2.3)",
    "ig_lora_scgpt": "scGPT LoRA (Integrated Gradients)",
    "attention_zero_shot_geneformer": "Geneformer zero-shot attention (pooled)",
    "attention_cssi_geneformer": "Geneformer CSSI-max attention",
    "attention_residualized_ols_geneformer": "Geneformer OLS-residual attention",
    "attention_lora_geneformer": "Geneformer LoRA (attention)",
}
METHOD_SHORT = {
    "attention_zero_shot_scgpt": "Pooled",
    "attention_cssi_scgpt": "CSSI",
    "attention_residualized_ols_scgpt": "OLS resid.",
    "ig_lora_scgpt": "IG LoRA",
    "attention_zero_shot_geneformer": "GF pooled",
    "attention_cssi_geneformer": "GF CSSI",
    "attention_residualized_ols_geneformer": "GF OLS resid.",
    "attention_lora_geneformer": "GF LoRA",
}
# Fixed categorical order (blue, orange, aqua, yellow, violet, brown, olive, magenta)
METHOD_COLORS = {
    "attention_zero_shot_scgpt": "#2a78d6",
    "attention_cssi_scgpt": "#eb6834",
    "attention_residualized_ols_scgpt": "#1baf7a",
    "ig_lora_scgpt": "#eda100",
    "attention_zero_shot_geneformer": "#7b5cd6",
    "attention_cssi_geneformer": "#a0672b",
    "attention_residualized_ols_geneformer": "#6f8f2f",
    "attention_lora_geneformer": "#d6458f",
}
# Dynamic-selection summary (n_selected per rule) of the methods that have one
SUMMARY_PATHS = {
    "attention_zero_shot_scgpt": "attention_zero_shot_scgpt/{ct}/csv/selection_summary.json",
    "attention_cssi_scgpt": "attention_zero_shot_scgpt/{ct}/cssi/csv/selection_summary.json",
    "attention_residualized_ols_scgpt": "attention_residualized_ols_scgpt/{ct}/csv/selection_summary.json",
    "attention_zero_shot_geneformer": "attention_zero_shot_geneformer/{ct}/csv/selection_summary.json",
    "attention_cssi_geneformer": "attention_zero_shot_geneformer/{ct}/cssi/csv/selection_summary.json",
    "attention_residualized_ols_geneformer": "attention_residualized_ols_geneformer/{ct}/csv/selection_summary.json",
}
CELL_TYPE_LABELS = {"astrocytes": "Astrocytes", "da_neurons": "DA neurons",
                    "microglia": "Microglia", "oligodendrocytes": "Oligodendrocytes"}
SELECTION_LABELS = {"top30": "static top-30", "elbow": "elbow (Kneedle)"}
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


def _method_legend(fig, methods, ncol=None) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in methods]
    fig.legend(handles, [METHOD_LABELS[m] for m in methods], loc="lower center",
               ncol=ncol or len(methods), frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _bar_offsets(n_methods: int):
    width = 0.8 / n_methods
    return width, [(i - (n_methods - 1) / 2) * width for i in range(n_methods)]


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
               color=METHOD_COLORS[method])

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


def plot_enrichment_summary(results_dir: Path, selection: str = "top30") -> Path:
    summary_dir = output_dir(results_dir, "enrichment_summary", selection)
    summary = pd.read_csv(summary_dir / "summary_table.csv")
    tests = pd.read_csv(summary_dir / "pairwise_tests.csv")
    methods = [m for m in methods_for(selection) if m in set(summary["method"])]

    panels = [
        ("n_significant_terms", None, "Number of significant\nGO/KEGG/REAC terms"),
        ("mean_information_content", "information_content", "Mean information content\n(significant terms)"),
        ("mean_intersection_size", "intersection_size", "Mean intersection size\n(genes per term)"),
        ("mean_functional_abundance", "functional_abundance", "Mean functional abundance\n(fraction of the term covered)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4.8))
    for ax, (metric, metric_key, title) in zip(axes, panels):
        _draw_enrichment_panel(ax, summary, tests, methods, metric, metric_key, title)
    fig.suptitle(f"g:Profiler enrichment - {SELECTION_LABELS[selection]} gene sets "
                 "(brackets: Mann-Whitney, BH-adjusted p)", fontsize=12)
    _method_legend(fig, methods)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, plot_path(results_dir, "enrichment_summary", selection))


# ---------------------------------------------------------------------------
# GWAS similarity (gene-level best match, from gene_matrix)
# ---------------------------------------------------------------------------
ONTOLOGIES = ["BP", "CC", "MF"]
DISTANCE = "SimRel"
DISEASE = "parkinson"

# Per-gene closest match (row max) to the PD reference, tagged by method;
# read straight from gene_matrix.py's matrix, not the pre-aggregated summary
def _bestmatch_per_gene(results_dir: Path, selection: str, cell_type: str, ontology: str) -> pd.DataFrame:
    path = (output_dir(results_dir, "gene_matrix", selection) / "matrices" / cell_type
            / f"{DISEASE}__{DISTANCE}__{ontology}.csv")
    mat = pd.read_csv(path, index_col=[0, 1])
    return mat.max(axis=1).rename("bestmatch").reset_index()

def _draw_gwas_panel(ax, results_dir: Path, selection: str, methods: list, ontology: str) -> None:
    data, positions, colors = [], [], []
    tick_positions, tick_labels = [], []
    pos = 0.0
    for cell_type in CELL_TYPES:
        df = _bestmatch_per_gene(results_dir, selection, cell_type, ontology)
        group_start = pos
        for method in methods:
            vals = df.loc[df["method"] == method, "bestmatch"].to_numpy()
            if len(vals) > 0:
                data.append(vals)
                positions.append(pos)
                colors.append(METHOD_COLORS[method])
            pos += 1.0
        tick_positions.append((group_start + pos - 1.0) / 2)
        tick_labels.append(CELL_TYPE_LABELS[cell_type])
        pos += 0.8  # gap between cell types

    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(SURFACE)
    for median in bp["medians"]:
        median.set_color(INK)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_title(f"GO:{ontology}", fontsize=11, fontweight="bold")
    _style(ax)

def plot_gwas_similarity(results_dir: Path, selection: str = "top30") -> Path:
    methods = methods_for(selection)
    fig, axes = plt.subplots(1, len(ONTOLOGIES), figsize=(15, 4.8), sharey=True)
    for ax, ontology in zip(axes, ONTOLOGIES):
        _draw_gwas_panel(ax, results_dir, selection, methods, ontology)
    axes[0].set_ylabel(f"Best-match similarity to PD GWAS genes\n({DISTANCE}, per gene)", fontsize=10)
    fig.suptitle(f"Closest PD GWAS gene per selected gene - {SELECTION_LABELS[selection]}", fontsize=12)
    _method_legend(fig, methods)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, plot_path(results_dir, "gwas_similarity", selection))


# ---------------------------------------------------------------------------
# GWAS set-level similarity (BMA, from gwas_similarity), PD vs control diseases
# ---------------------------------------------------------------------------
def plot_gwas_set_similarity(results_dir: Path, selection: str = "top30") -> Path:
    df = pd.read_csv(output_dir(results_dir, "gwas_similarity", selection) / "gene_level_similarity_all.csv")
    df = df[df["distance"] == DISTANCE]
    methods = [m for m in methods_for(selection) if m in set(df["method"])]
    x = np.arange(len(CELL_TYPES))
    width, offsets = _bar_offsets(len(methods))

    fig, axes = plt.subplots(1, len(ONTOLOGIES), figsize=(15, 4.8), sharey=True)
    for ax, ontology in zip(axes, ONTOLOGIES):
        sub = df[df["go_source"] == f"GO:{ontology}"].set_index(["reference", "method", "cell_type"])["similarity"]
        for method, off in zip(methods, offsets):
            pd_vals = [sub.get(("parkinson", method, c), np.nan) for c in CELL_TYPES]
            ax.bar(x + off, pd_vals, width=width * 0.92, color=METHOD_COLORS[method])
            # Control diseases: the PD bar only means something relative to these
            for ref, marker in [("alzheimer", "o"), ("cad", "x")]:
                vals = [sub.get((ref, method, c), np.nan) for c in CELL_TYPES]
                ax.scatter(x + off, vals, marker=marker, s=22, color=INK, zorder=3,
                           facecolors="none" if marker == "o" else INK, linewidths=1)
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_TYPE_LABELS[c] for c in CELL_TYPES], rotation=20, ha="right")
        ax.set_title(f"GO:{ontology}", fontsize=11, fontweight="bold")
        _style(ax)
    axes[0].set_ylabel(f"Gene-set similarity to GWAS genes\n({DISTANCE}, BMA)", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in methods]
    handles += [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=INK),
                plt.Line2D([], [], ls="", marker="x", color=INK)]
    labels = [f"{METHOD_LABELS[m]} vs PD (bar)" for m in methods] + ["vs Alzheimer (control)", "vs CAD (control)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"GWAS gene-set similarity - {SELECTION_LABELS[selection]}", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, plot_path(results_dir, "gwas_set_similarity", selection))


# ---------------------------------------------------------------------------
# Gene-set sizes and overlap between methods
# ---------------------------------------------------------------------------
def _jaccard(a: list, b: list) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def plot_gene_set_overlap(results_dir: Path, selection: str = "top30") -> Path:
    methods = methods_for(selection)
    fig, axes = plt.subplots(1, len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 4.2))
    for ax, cell_type in zip(axes, CELL_TYPES):
        sets = {m: load_genes(results_dir, m, selection, cell_type) for m in methods}
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
    fig.suptitle(f"Gene-set overlap between methods - {SELECTION_LABELS[selection]} "
                 "(Jaccard index, shared genes in brackets)", fontsize=12)
    fig.tight_layout()
    return _save(fig, plot_path(results_dir, "gene_set_overlap", selection))


def plot_gene_set_sizes(results_dir: Path) -> Path:
    """Number of genes each method selects per selection rule (all dynamic rules)."""
    rules = ["top30", "z-score", "elbow"]
    sources = SUMMARY_PATHS
    methods = list(sources)
    fig, axes = plt.subplots(1, len(rules), figsize=(4 * len(rules), 4.2), sharey=True)
    x = np.arange(len(CELL_TYPES))
    width, offsets = _bar_offsets(len(methods))
    for ax, rule in zip(axes, rules):
        for method, off in zip(methods, offsets):
            vals = []
            for cell_type in CELL_TYPES:
                path = results_dir / sources[method].format(ct=cell_type)
                n = np.nan
                if path.exists():
                    n = json.loads(path.read_text()).get(rule, {}).get("n_selected", np.nan)
                vals.append(n)
            bars = ax.bar(x + off, vals, width=width * 0.92, color=METHOD_COLORS[method])
            ax.bar_label(bars, labels=["" if pd.isna(v) else int(v) for v in vals],
                         fontsize=7, color=INK_2, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_TYPE_LABELS[c] for c in CELL_TYPES], rotation=20, ha="right")
        ax.set_title(rule, fontsize=11, fontweight="bold")
        _style(ax)
    axes[0].set_ylabel("Genes selected")
    fig.suptitle("Gene-set size per selection rule", fontsize=12)
    _method_legend(fig, methods)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, results_dir / "plots" / "gene_set_sizes.png")


# ---------------------------------------------------------------------------
# Dynamic cut-offs on each attention method's ranked score curve
# ---------------------------------------------------------------------------
# All-gene score files (rank, gene, score). Pooled attention has no dedicated
# all-gene file in older runs, so the permutation audit (one row per testable
# gene) is used as fallback.
ALL_SCORE_PATHS = {
    "attention_zero_shot_scgpt": ["attention_zero_shot_scgpt/{ct}/csv/all_genes.csv",
                                  "attention_zero_shot_scgpt/{ct}/csv/permutation_audit.csv"],
    "attention_cssi_scgpt": ["attention_zero_shot_scgpt/{ct}/cssi_genes.csv"],
    "attention_residualized_ols_scgpt": ["attention_residualized_ols_scgpt/{ct}/csv/ols.csv"],
    "attention_zero_shot_geneformer": ["attention_zero_shot_geneformer/{ct}/csv/all_genes.csv"],
    "attention_cssi_geneformer": ["attention_zero_shot_geneformer/{ct}/cssi_genes.csv"],
    "attention_residualized_ols_geneformer": ["attention_residualized_ols_geneformer/{ct}/csv/ols.csv"],
}
CUTOFF_STYLES = {"top30": ":", "z-score": "--", "elbow": "-"}
MAX_RANK_SHOWN = 200


def _all_scores(results_dir: Path, method: str, cell_type: str):
    for rel in ALL_SCORE_PATHS[method]:
        path = results_dir / rel.format(ct=cell_type)
        if path.exists():
            return np.sort(pd.read_csv(path)["score"].to_numpy(dtype=float))[::-1]
    return None


def plot_dynamic_selection(results_dir: Path) -> Path:
    methods = list(ALL_SCORE_PATHS)
    fig, axes = plt.subplots(len(methods), len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 3.2 * len(methods)))
    for r, method in enumerate(methods):
        for c, cell_type in enumerate(CELL_TYPES):
            ax = axes[r, c]
            scores = _all_scores(results_dir, method, cell_type)
            summary_path = results_dir / SUMMARY_PATHS[method].format(ct=cell_type)
            if scores is None or not summary_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                ax.set_axis_off()
                continue
            summary = json.loads(summary_path.read_text())
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
    fig.suptitle("Ranked score curves and dynamic cut-offs per attention method", fontsize=12)
    fig.tight_layout()
    return _save(fig, results_dir / "plots" / "dynamic_selection_cutoffs.png")


# ---------------------------------------------------------------------------
# Method 2.3 diagnostics: what OLS removes from the attention score
# ---------------------------------------------------------------------------
def plot_ols_residualization(results_dir: Path) -> Path:
    fig, axes = plt.subplots(2, len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 7.5))
    for c, cell_type in enumerate(CELL_TYPES):
        base = results_dir / "attention_residualized_ols_scgpt" / cell_type
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
                       color=METHOD_COLORS["attention_zero_shot_scgpt"], edgecolors=SURFACE, lw=0.6,
                       label="top-30 raw only")
            ax.scatter(df.loc[only_res, "attention_score"], df.loc[only_res, ycol], s=18,
                       color=METHOD_COLORS["attention_residualized_ols_scgpt"], edgecolors=SURFACE, lw=0.6,
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
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Method 2.3: attention explained by expression mean/variance/dropout (top) "
                 "vs residual kept for ranking (bottom; dashed y = x)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, results_dir / "plots" / "ols_residualization.png")


# ---------------------------------------------------------------------------
# Static top-30 vs elbow, per method: set size, enrichment and PD similarity
# ---------------------------------------------------------------------------
SELECTION_HATCH = {"top30": None, "elbow": "////"}
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


def plot_selection_comparison(results_dir: Path, selections=SELECTIONS) -> Path:
    df = pd.concat([_selection_metrics(results_dir, s) for s in selections], ignore_index=True)
    df.to_csv(results_dir / "plots" / "selection_comparison.csv", index=False)
    methods = [m for m in METHODS_ORDER if m in set(df["method"])]
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
            ax.set_xticklabels([METHOD_SHORT[m] for m in methods], fontsize=8)
            if r == 0:
                ax.set_title(CELL_TYPE_LABELS[cell_type], fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            _style(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    for selection in selections:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK_2,
                                     hatch=SELECTION_HATCH[selection]))
        labels.append(SELECTION_LABELS[selection])
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Static top-30 (solid) vs elbow (hatched) gene sets per method "
                 "(IG LoRA has no elbow set)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, results_dir / "plots" / "selection_comparison.png")


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
