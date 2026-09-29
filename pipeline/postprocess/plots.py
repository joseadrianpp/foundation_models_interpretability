from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .enrichment import CELL_TYPES, METHODS

METHOD_LABELS = {"attention_zero_shot_scgpt": "scGPT zero-shot (attention)",
                 "ig_lora_scgpt": "scGPT LoRA (Integrated Gradients)",
                 "attention_zero_shot_geneformer": "Geneformer zero-shot (attention)",
                 "attention_lora_geneformer": "Geneformer LoRA (attention)"}
METHOD_COLORS = {"attention_zero_shot_scgpt": "#4878CF", "ig_lora_scgpt": "#D65F5F",
                 "attention_zero_shot_geneformer": "#6ACC65", "attention_lora_geneformer": "#B47CC7"}
CELL_TYPE_LABELS = {"astrocytes": "Astrocytes", "da_neurons": "DA neurons",
                     "microglia": "Microglia", "oligodendrocytes": "Oligodendrocytes"}
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
# The bars of all methods share 0.8 of the space of each cell type
BAR_WIDTH = 0.8 / len(METHODS)

def _stars_for(p: float) -> str:
    for threshold, label in SIG_LEVELS:
        if p < threshold:
            return label
    return ""

# One bracket per significant pair of methods, each one above the previous one.
# Returns the height of the last bracket
def _draw_brackets(ax, rows: pd.DataFrame, x_ct: float, offsets: dict, y: float, step: float) -> float:
    for _, row in rows.iterrows():
        y += step
        x1, x2 = x_ct + offsets[row["method_a"]], x_ct + offsets[row["method_b"]]
        ax.plot([x1, x1, x2, x2], [y - step * 0.3, y, y, y - step * 0.3], color="black", lw=1)
        ax.text((x1 + x2) / 2, y, _stars_for(row["p_value"]), ha="center", va="bottom", fontsize=9)
    return y

def _draw_enrichment_panel(ax, summary: pd.DataFrame, tests: pd.DataFrame,
                            metric: str, metric_key: str, title: str) -> None:
    x = np.arange(len(CELL_TYPES))
    # Position of each method's bar with respect to the cell type tick
    offsets = {m: (i - (len(METHODS) - 1) / 2) * BAR_WIDTH for i, m in enumerate(METHODS)}
    tops = {}
    for method in METHODS:
        heights = []
        for cell_type in CELL_TYPES:
            row = summary[(summary["cell_type"] == cell_type) & (summary["method"] == method)]
            val = row[metric].iloc[0] if len(row) else 0.0
            val = 0.0 if pd.isna(val) else val
            heights.append(val)
            tops[(cell_type, method)] = val
        ax.bar(x + offsets[method], heights, width=BAR_WIDTH, label=METHOD_LABELS[method], color=METHOD_COLORS[method])

    # n_significant_terms has no pairwise test (only IC/intersection_size/functional abundance do)
    ymax = max(tops.values(), default=0.0)
    if metric_key is not None and ymax > 0:
        step = ymax * 0.14
        top = ymax
        for i, cell_type in enumerate(CELL_TYPES):
            rows = tests[(tests["cell_type"] == cell_type) & (tests["metric"] == metric_key)
                         & (tests["p_value"] < 0.05)]
            y = max(tops[(cell_type, m)] for m in METHODS)
            top = max(top, _draw_brackets(ax, rows, x[i], offsets, y, step))
        ax.set_ylim(top=max(ymax * 1.25, top + step))

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_TYPE_LABELS[c] for c in CELL_TYPES], rotation=20, ha="right")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

def plot_enrichment_summary(results_dir: Path) -> Path:
    summary_dir = results_dir / "enrichment_summary"
    summary = pd.read_csv(summary_dir / "summary_table.csv")
    tests = pd.read_csv(summary_dir / "pairwise_tests.csv")

    panels = [
        ("n_significant_terms", None, "Number of significant\nGO/KEGG/REAC terms"),
        ("mean_information_content", "information_content", "Mean information content\n(significant terms)"),
        ("mean_intersection_size", "intersection_size", "Mean intersection size\n(genes per term)"),
        ("mean_functional_abundance", "functional_abundance", "Mean functional abundance\n(fraction of the term covered)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.5))
    for ax, (metric, metric_key, title) in zip(axes, panels):
        _draw_enrichment_panel(ax, summary, tests, metric, metric_key, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = results_dir / "plots" / "enrichment_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path

ONTOLOGIES = ["BP", "CC", "MF"]
DISTANCE = "SimRel"
DISEASE = "parkinson"

# Per-gene closest match (row max) to the PD reference, tagged by method;
# read straight from gene_matrix.py's matrix, not the pre-aggregated summary
def _bestmatch_per_gene(results_dir: Path, cell_type: str, ontology: str) -> pd.DataFrame:
    path = results_dir / "gene_matrix" / "matrices" / cell_type / f"{DISEASE}__{DISTANCE}__{ontology}.csv"
    mat = pd.read_csv(path, index_col=[0, 1])
    return mat.max(axis=1).rename("bestmatch").reset_index()

def _draw_gwas_panel(ax, results_dir: Path, ontology: str) -> None:
    data, positions, colors = [], [], []
    tick_positions, tick_labels = [], []
    pos = 0.0
    for cell_type in CELL_TYPES:
        df = _bestmatch_per_gene(results_dir, cell_type, ontology)
        group_start = pos
        for method in METHODS:
            vals = df.loc[df["method"] == method, "bestmatch"].to_numpy()
            if len(vals) == 0:
                pos += 1.0
                continue
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
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_title(f"GO:{ontology}", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

def plot_gwas_similarity(results_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(ONTOLOGIES), figsize=(14, 4.5), sharey=True)
    for ax, ontology in zip(axes, ONTOLOGIES):
        _draw_gwas_panel(ax, results_dir, ontology)
    axes[0].set_ylabel(f"Best-match similarity to PD\n({DISTANCE}, BMA)", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in METHODS]
    fig.legend(handles, [METHOD_LABELS[m] for m in METHODS], loc="lower center",
               ncol=len(METHODS), frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = results_dir / "plots" / "gwas_similarity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path