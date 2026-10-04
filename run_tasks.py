#!/usr/bin/env python
"""
Interpretability outputs for the scGPT and Geneformer models:

  1. scGPT zero-shot CLS-attention signed score (PD vs Control) -> top-30 genes,
     plus the CSSI-max score over Leiden strata and the dynamic selections
     (top30 / z-score / elbow) of both.
  2. Integrated Gradients on the LoRA fine-tuned scGPT model -> top-30 genes.
  3. The same outputs as 1 (pooled + CSSI + dynamic selections) with the
     zero-shot Geneformer model.
  4. The same attention score with the LoRA fine-tuned Geneformer model, only
     on its correctly classified cells.
  5. OLS residualization of the zero-shot attention of 1 and 3 (scGPT and
     Geneformer) -> all genes + selections.

Runs on the full dataset (train + val + test combined), on Dataset A or
Dataset B (--dataset), each with its own results folder.

GPU-bound model tasks (1-4) can use --gpus to run independent cell-type
workers on multiple GPUs. The OLS tasks (5) run on CPU from the attention
tasks' output, so they need 1 / 3 to have run first with the same
--max-cells-per-type.
"""

import argparse
import csv
import json
import sys
from multiprocessing import get_context
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from datasets import concatenate_datasets, load_from_disk

from pipeline import config as C
from pipeline.postprocess.enrichment import task_enrichment
from pipeline.postprocess.enrichment_summary import task_enrichment_summary
from pipeline.postprocess.gene_matrix import task_gene_matrix
from pipeline.postprocess.gene_sets import SELECTIONS
from pipeline.postprocess.gwas_similarity import DEFAULT_DISTANCES as GO3_DISTANCES, task_gwas_similarity
from pipeline.postprocess.plots import task_plots
from pipeline.steps import _step7_common as common
from pipeline.steps import _step7_geneformer as gf
from pipeline.steps import attention_cssi_leiden as cssi
from pipeline.steps import attention_dynamic_selection as dynamic
from pipeline.steps import step7_attention_residualized_ols as residualized_ols
from pipeline.steps import step7_explainability_ig_scgpt as ig
from pipeline.utils import progress
from pipeline.utils.geneformer_model import load_gene_names, load_token_dictionary
from pipeline.utils.scgpt_model import load_model_configs, load_vocab


DATA_DIR = _HERE / "data"
# The two datasets have the same four cell types and the same kind of split,
# and each folder holds its own cells and LoRA models
DATASET_DIRS = {"A": DATA_DIR / "dataset_a", "B": DATA_DIR / "dataset_b"}

# scGPT files inside a dataset folder
SCGPT_H5ADS = [Path("mixed_split_scgpt") / f"Mixed_HVG_1000_pc_noXY_noMT_mixed_{split}.h5ad"
               for split in ("train", "val", "test")]
LORA_FULL_MODEL_SCGPT = Path("step5_1_lora_scgpt") / "best_full_model.pt"

# Same cells and splits for Geneformer, already tokenized (Hugging Face datasets)
GENEFORMER_DATASETS = [Path("mixed_split_geneformer") / f"Mixed_pc_noXY_noMT_mixed_{split}.dataset"
                       for split in ("train", "val", "test")]
LORA_DIR_GENEFORMER = Path("step5_1_lora_geneformer")

TOP_N = 30
SEED = 42
CSSI_N_NEIGHBORS = 15
CSSI_TARGET_K = 6
CSSI_MIN_CELLS_PER_CONDITION = 20
# BH over ~1000 genes needs >= m/alpha = 20k permutations for a single gene to
# reach q < 0.05 (min q = m / (n_perm + 1)); 50k gives margin (min q ~ 0.02).
DYNAMIC_PERMUTATIONS = dynamic.PERMUTATION_N
DYNAMIC_PERMUTATION_ALPHA = 0.05
DYNAMIC_PERMUTATION_EFFECT_SIZE = 0.5


def _signed_score(mean_pd: float, mean_ctrl: float) -> float:
    return mean_pd - mean_ctrl


def _write_top_n(score_map: dict, out_csv: Path, n: int = TOP_N) -> None:
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene", "score"])
        for r, (g, s) in enumerate(ranked[:n], start=1):
            w.writerow([r, g, s])


def _write_all_scores(score_map: dict, out_csv: Path) -> None:
    """Write all genes ranked by signed score (used for CSSI output)."""
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene", "score"])
        for r, (g, s) in enumerate(ranked, start=1):
            w.writerow([r, g, s])


def _load_full_dataset_scgpt(data_dir: Path) -> ad.AnnData:
    return ad.concat([sc.read_h5ad(data_dir / p) for p in SCGPT_H5ADS])


def _load_full_dataset_geneformer(data_dir: Path):
    return concatenate_datasets([load_from_disk(str(data_dir / p)) for p in GENEFORMER_DATASETS])


# Default task options; main() overrides them from the command line
DEFAULT_OPTS = {"donor_col": None, "permutation": False, "max_cells_per_type": None,
                "batch_scale": "auto"}

# The model batch sizes (scGPT attention 16, Geneformer attention 8, IG 8 cells
# in chunks of 32) fit an 8 GB GPU. --batch-scale multiplies all of them;
# "auto" takes one step per GPU_GIB_PER_SCALE of GPU memory (RTX 2070 x1,
# L40 x4, H100 x7, H200 x8), capped at MAX_BATCH_SCALE
GPU_GIB_PER_SCALE = 10
MAX_BATCH_SCALE = 8


def _batch_scale(device: torch.device, opts: dict) -> int:
    scale = opts["batch_scale"]
    if scale != "auto":
        return max(1, int(scale))
    if device.type != "cuda":
        return 1
    gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    return int(max(1, min(MAX_BATCH_SCALE, gib // GPU_GIB_PER_SCALE)))


# Indices of one cell type, optionally a seeded random subset of at most
# `max_cells` (for quick test runs). The seed is fixed per call, so the subset
# does not depend on which cell types were processed before
def _cell_type_indices(values: np.ndarray, ct: str, max_cells=None) -> np.ndarray:
    idx = np.where(values == ct)[0]
    if max_cells is not None and idx.size > max_cells:
        idx = np.sort(np.random.default_rng(SEED).choice(idx, max_cells, replace=False))
    return idx


def _cell_type_subsets_scgpt(adata, allowed_cell_types=None, max_cells=None):
    """Yield (cell type, slug, AnnData subset), optionally restricted to assigned cell types."""
    values = adata.obs[C.CELL_TYPE_COL].astype(str).to_numpy()
    allowed = None if allowed_cell_types is None else set(allowed_cell_types)

    for ct in sorted(set(values)):
        if allowed is not None and ct not in allowed:
            continue
        yield ct, common._slug_cell_type(ct), adata[_cell_type_indices(values, ct, max_cells)].copy()


def _cell_type_subsets_geneformer(ds, allowed_cell_types=None, max_cells=None):
    """Same as _cell_type_subsets_scgpt but for a Hugging Face dataset."""
    values = np.array(ds[C.CELL_TYPE_COL])
    allowed = None if allowed_cell_types is None else set(allowed_cell_types)

    for ct in sorted(set(values)):
        if allowed is not None and ct not in allowed:
            continue
        yield ct, common._slug_cell_type(ct), ds.select(_cell_type_indices(values, ct, max_cells))


def _cell_type_counts(task_name: str, data_dir: Path, max_cells=None) -> dict:
    """Cells per cell type, read without loading expression values or tokens."""
    counts = {}
    if task_name.endswith("geneformer"):
        columns = [load_from_disk(str(data_dir / p))[C.CELL_TYPE_COL] for p in GENEFORMER_DATASETS]
    else:
        columns = [ad.read_h5ad(data_dir / p, backed="r").obs[C.CELL_TYPE_COL].astype(str)
                   for p in SCGPT_H5ADS]
    for col in columns:
        for ct, n in pd.Series(col).value_counts().items():
            counts[str(ct)] = counts.get(str(ct), 0) + int(n)
    if max_cells is not None:
        counts = {ct: min(n, max_cells) for ct, n in counts.items()}
    return counts


def _split_cell_types(counts: dict, n_workers: int):
    """
    Greedily balance cell types by number of cells.

    A whole cell type stays on one GPU. This avoids merging partial results and
    guarantees that each output CSV is produced by exactly one worker.
    """
    workers = [[] for _ in range(n_workers)]
    loads = [0] * n_workers

    for ct, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        worker = min(range(n_workers), key=loads.__getitem__)
        workers[worker].append(ct)
        loads[worker] += count

    return workers, loads


def _saved_strata(out_dir: Path, embeddings: np.ndarray, embedding_source: str):
    """(strata, metadata) an earlier run left in `out_dir` for these exact embeddings, or None.

    Leiden is deterministic, so with bit-identical embeddings and the same
    Leiden settings the strata of a rerun are the saved ones; reusing them
    skips the sweep (an hour on Dataset B oligodendrocytes).
    """
    try:
        saved = np.load(out_dir / "cell_embeddings.npy")
        meta = json.loads((out_dir / "cssi_metadata.json").read_text())
        strata = pd.read_csv(out_dir / "leiden_clusters.csv")["leiden"].to_numpy()
    except (FileNotFoundError, ValueError, KeyError):
        return None
    sweep = meta.get("resolution_sweep")
    if not (sweep and saved.shape == embeddings.shape and np.array_equal(saved, embeddings)
            and strata.size == len(embeddings)
            and meta.get("n_neighbors") == min(CSSI_N_NEIGHBORS, len(embeddings) - 1)
            and meta.get("target_k") == CSSI_TARGET_K and meta.get("embedding_source") == embedding_source
            and meta.get("leiden_patience", cssi.LEIDEN_PATIENCE) == cssi.LEIDEN_PATIENCE
            and np.allclose([r for r, _ in sweep], cssi.DEFAULT_RESOLUTIONS[:len(sweep)])):
        return None
    print("  Leiden: same embeddings as the saved run, reusing its strata", flush=True)
    keys = ["n_neighbors", "resolution", "n_clusters", "target_k", "metric", "embedding_source",
            "resolution_sweep"]
    return strata, {**{k: meta[k] for k in keys}, "leiden_patience": cssi.LEIDEN_PATIENCE}


def _write_attention_outputs(cache: cssi.AttentionCache, out_dir: Path, donor_ids=None,
                             cell_mask=None, save_cells: bool = False) -> None:
    """Pooled + CSSI scores, dynamic selections and Leiden artifacts of one cell type.

    Model independent: it only reads the AttentionCache. With `donor_ids`
    the donor-level permutation selector also runs (--permutation). With
    `cell_mask` the scores only use those cells (the correctly classified
    ones of a fine-tuned classifier); Leiden still clusters every cell. With
    `save_cells` the per-cell token scores go to cell_scores.npz, so CSSI can
    be recomputed without the model.
    """
    means, cssi_scores, gate, strata, meta = cssi.compute_attention_cssi(
        cache,
        n_neighbors=CSSI_N_NEIGHBORS,
        target_k=CSSI_TARGET_K,
        min_cells_per_condition=CSSI_MIN_CELLS_PER_CONDITION,
        random_state=SEED,
        desc=f"{out_dir.parent.name} {out_dir.name} Leiden",
        cell_mask=cell_mask,
        precomputed_strata=_saved_strata(out_dir, cache.embeddings, cache.embedding_source),
    )
    if not gate["ok"]:
        print(f"  skipped: {gate['reason']}", flush=True)
        return

    pooled_scores = common._gene_scores(means, _signed_score)
    donor_data = None
    if donor_ids is not None:
        donor_data = dynamic.donor_attention_data(cache, donor_ids, strata)
        # The OLS task residualizes this matrix for its own permutation selector
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "donor_attention.npz", donors=np.asarray(donor_data["donors"]),
                            labels=donor_data["labels"], genes=np.asarray(donor_data["genes"]),
                            scores=donor_data["scores"])
    _write_score_outputs(pooled_scores, cssi_scores, strata, meta, cache.embeddings, out_dir, donor_data)
    if save_cells:
        cache.save(out_dir / "cell_scores.npz")


def _write_score_outputs(pooled_scores: dict, cssi_scores: dict, strata: np.ndarray, meta: dict,
                         embeddings: np.ndarray, out_dir: Path, donor_data=None) -> None:
    """Write one cell type's pooled and CSSI gene scores, their selections and the Leiden artifacts.

    Shared by every score (attention, LoRA attention, IG): top30_genes.csv,
    csv/all_genes.csv and csv/<selection>.csv (pooled), cssi_genes.csv and
    cssi/csv/<selection>.csv (CSSI), cell_embeddings.npy, leiden_clusters.csv
    and cssi_metadata.json.
    """
    print(
        f"  CSSI: {meta.get('n_clusters', 1)} Leiden strata "
        f"(resolution={meta.get('resolution', 0):.4g}), "
        f"{embeddings.shape[1]}-D embeddings",
        flush=True,
    )

    # Preserve the original baseline/output consumed by downstream tasks.
    _write_top_n(pooled_scores, out_dir / "top30_genes.csv")
    _write_all_scores(cssi_scores, out_dir / "cssi_genes.csv")
    _write_all_scores(pooled_scores, out_dir / "csv" / "all_genes.csv")

    # Dynamic top-N: pooled attention and CSSI separately. Permutation is
    # donor-level and therefore uses the same donor identities in both.
    summaries = {}
    for name, scores, sel_dir, cssi_mode in [("pooled", pooled_scores, out_dir / "csv", False),
                                             ("CSSI", cssi_scores, out_dir / "cssi" / "csv", True)]:
        summaries[name] = dynamic.write_selection_family(
            scores,
            sel_dir,
            donor_data=donor_data,
            cssi_mode=cssi_mode,
            random_state=SEED,
            permutation_n=DYNAMIC_PERMUTATIONS,
            permutation_alpha=DYNAMIC_PERMUTATION_ALPHA,
            permutation_effect_size=DYNAMIC_PERMUTATION_EFFECT_SIZE,
        )

    np.save(out_dir / "cell_embeddings.npy", embeddings)
    pd.DataFrame({"cell_index": np.arange(len(strata)), "leiden": strata}).to_csv(
        out_dir / "leiden_clusters.csv", index=False
    )
    cssi.save_metadata(out_dir / "cssi_metadata.json", meta)

    print(f"  wrote {out_dir / 'top30_genes.csv'}", flush=True)
    print(f"  wrote {out_dir / 'cssi_genes.csv'} ({len(cssi_scores)} genes)", flush=True)
    for name, summary in summaries.items():
        print(f"  {name} dynamic N: " + ", ".join(f"{k}={v['n_selected']}" for k, v in summary.items()),
              flush=True)


def task_attention_scgpt(data_dir: Path, results_dir: Path, device: torch.device,
                         cell_types=None, opts=None) -> None:
    opts = {**DEFAULT_OPTS, **(opts or {})}
    scale = _batch_scale(device, opts)
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset_scgpt(data_dir)
    model = common._load_zero_shot_model(vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets_scgpt(adata, cell_types, opts["max_cells_per_type"]):
        print(f"\n# attention_scgpt | {ct} | {sub.n_obs:,} cells | {device} | batch x{scale} #", flush=True)
        pt = common._tokenise(sub, vocab)

        donor_ids = None
        if opts["permutation"]:
            donor_column = dynamic.resolve_donor_column(sub, opts["donor_col"])
            donor_ids = sub.obs[donor_column].astype(str).to_numpy()
            print(f"  donor column: {donor_column} | {np.unique(donor_ids).size} donors", flush=True)

        # One forward pass; pooled, CSSI and donor-level scores all read this cache
        cache = cssi.collect_scgpt(model, pt, vocab, model_configs["nheads"], device, n_cls=2,
                                   batch_size=cssi.SCGPT_ATTN_BATCH_SIZE * scale,
                                   desc=f"attention_scgpt {slug} [{device}]")
        _write_attention_outputs(cache, results_dir / "attention_zero_shot_scgpt" / slug, donor_ids)

    del model
    torch.cuda.empty_cache()


def task_ig_lora_scgpt(data_dir: Path, results_dir: Path, device: torch.device,
                       cell_types=None, opts=None) -> None:
    opts = {**DEFAULT_OPTS, **(opts or {})}
    scale = _batch_scale(device, opts)
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset_scgpt(data_dir)

    print(f"[{device}] loading LoRA model...", flush=True)
    model = common._load_lora_model(data_dir / LORA_FULL_MODEL_SCGPT, vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets_scgpt(adata, cell_types, opts["max_cells_per_type"]):
        print(f"\n=== ig_lora_scgpt | {ct} | {sub.n_obs:,} cells | {device} | batch x{scale} ===", flush=True)

        pt = common._tokenise(sub, vocab)

        result = ig._compute_ig_result(
            model, pt, vocab, device, batch_size=ig.IG_BATCH_SIZE * scale,
            internal_batch_size=ig.IG_INTERNAL_BATCH_SIZE * scale,
            desc=f"ig_lora_scgpt {slug} [{device}]")

        if not result.gate["ok"]:
            print(f"  skipped: {result.gate['reason']}", flush=True)
            continue

        # CSSI: Leiden strata of the LoRA model's own cell embedding, each
        # gene scored by its largest mean |IG| over the valid strata
        out_dir = results_dir / "ig_lora_scgpt" / slug
        source = f"scGPT LoRA residual-stream CLS embedding entering transformer layer {common.NUM_ATTN_LAYERS}"
        saved = _saved_strata(out_dir, result.embeddings, source)
        strata, meta = saved if saved is not None else cssi._leiden_from_embeddings(
            result.embeddings, n_neighbors=CSSI_N_NEIGHBORS, target_k=CSSI_TARGET_K, random_state=SEED,
            embedding_source=source, desc=f"ig_lora_scgpt {slug} Leiden")
        cssi_scores, valid_strata = ig.stratified_ig_scores(result, strata, vocab, CSSI_MIN_CELLS_PER_CONDITION)
        meta.update(valid_strata=valid_strata, n_cells=int(result.labels.size),
                    n_cells_scored=int(result.correct.sum()), n_embedding_dims=int(result.embeddings.shape[1]))
        _write_score_outputs(result.mean_abs, cssi_scores, strata, meta, result.embeddings, out_dir)
        # Per-cell |IG| of every token, so CSSI can be recomputed without the model
        result.save(out_dir / "cell_scores.npz")

    del model
    torch.cuda.empty_cache()


def task_attention_geneformer(data_dir: Path, results_dir: Path, device: torch.device,
                              cell_types=None, opts=None) -> None:
    # Like the scGPT attention task: every cell, grouped by its true label,
    # with the same pooled + CSSI scores and dynamic selections
    opts = {**DEFAULT_OPTS, **(opts or {})}
    scale = _batch_scale(device, opts)
    if opts["permutation"]:
        print("  --permutation ignored: the Geneformer datasets have no donor column", flush=True)
    token_dict, gene_names = load_token_dictionary(), load_gene_names(data_dir)
    ds = _load_full_dataset_geneformer(data_dir)
    model = gf._load_zero_shot_model(device)

    for ct, slug, sub in _cell_type_subsets_geneformer(ds, cell_types, opts["max_cells_per_type"]):
        print(f"\n# attention_geneformer | {ct} | {sub.num_rows:,} cells | {device} | batch x{scale} #",
              flush=True)
        cache = cssi.collect_geneformer(model, gf._to_inputs(sub), token_dict, gene_names, device,
                                        batch_size=gf.ATTN_BATCH_SIZE * scale,
                                        desc=f"attention_geneformer {slug} [{device}]")
        _write_attention_outputs(cache, results_dir / "attention_zero_shot_geneformer" / slug)

    del model
    torch.cuda.empty_cache()


def task_attention_lora_geneformer(data_dir: Path, results_dir: Path, device: torch.device,
                                   cell_types=None, opts=None) -> None:
    # Attention scores of the LoRA classifier (pooled, CSSI, selections) only
    # on the cells it classifies correctly (the same convention as upstream)
    opts = {**DEFAULT_OPTS, **(opts or {})}
    scale = _batch_scale(device, opts)
    token_dict, gene_names = load_token_dictionary(), load_gene_names(data_dir)
    ds = _load_full_dataset_geneformer(data_dir)
    model = gf._load_lora_model(data_dir / LORA_DIR_GENEFORMER, device)

    for ct, slug, sub in _cell_type_subsets_geneformer(ds, cell_types, opts["max_cells_per_type"]):
        print(f"\n# attention_lora_geneformer | {ct} | {sub.num_rows:,} cells | {device} | batch x{scale} #",
              flush=True)
        cache = cssi.collect_geneformer(model, gf._to_inputs(sub), token_dict, gene_names, device,
                                        batch_size=gf.ATTN_BATCH_SIZE * scale, use_predictions=True,
                                        desc=f"attention_lora_geneformer {slug} [{device}]")
        n_correct = {c: int(np.sum(cache.correct & (cache.labels == c))) for c in (0, 1)}
        if min(n_correct.values()) == 0:
            print(f"  skipped: correctly-classified counts: control={n_correct[0]}, pd={n_correct[1]}",
                  flush=True)
            continue
        # Pooled, CSSI (strata of the LoRA model's own embedding) and selections
        _write_attention_outputs(cache, results_dir / "attention_lora_geneformer" / slug, cell_mask=cache.correct,
                                 save_cells=True)

    del model
    torch.cuda.empty_cache()


# OLS residualization of the attention task output (CPU, no model). The cells
# (and --max-cells-per-type subset) are the same the attention task used
def task_attention_residualized_ols_scgpt(data_dir: Path, results_dir: Path, opts=None,
                                          source="attention_zero_shot_scgpt",
                                          target="attention_residualized_ols_scgpt") -> None:
    opts = {**DEFAULT_OPTS, **(opts or {})}
    vocab = load_vocab()
    adata = _load_full_dataset_scgpt(data_dir)
    for ct, slug, sub in _cell_type_subsets_scgpt(adata, None, opts["max_cells_per_type"]):
        print(f"\n# {target} | {ct} | {sub.n_obs:,} cells #", flush=True)
        genes, features = residualized_ols.scgpt_gene_features(sub, vocab)
        residualized_ols.residualize_cell_type(results_dir, "scgpt", slug, genes, features,
                                               sub.n_obs, opts["permutation"], source=source, target=target)


def task_attention_residualized_ols_geneformer(data_dir: Path, results_dir: Path, opts=None,
                                               source="attention_zero_shot_geneformer",
                                               target="attention_residualized_ols_geneformer") -> None:
    opts = {**DEFAULT_OPTS, **(opts or {})}
    token_dict, gene_names = load_token_dictionary(), load_gene_names(data_dir)
    ds = _load_full_dataset_geneformer(data_dir)
    for ct, slug, sub in _cell_type_subsets_geneformer(ds, None, opts["max_cells_per_type"]):
        print(f"\n# {target} | {ct} | {sub.num_rows:,} cells #", flush=True)
        genes, features = residualized_ols.geneformer_rank_features(
            gf._to_inputs(sub)["input_ids"], token_dict, gene_names)
        residualized_ols.residualize_cell_type(results_dir, "geneformer", slug, genes, features,
                                               sub.num_rows, opts["permutation"], source=source, target=target)


# The same OLS for the LoRA scores: IG of the LoRA scGPT model (expression
# features from the h5ad) and attention of the LoRA Geneformer model (rank
# pseudo-expression). The features cover every cell of the cell type
def task_ig_residualized_ols_lora_scgpt(data_dir: Path, results_dir: Path, opts=None) -> None:
    task_attention_residualized_ols_scgpt(data_dir, results_dir, opts, source="ig_lora_scgpt",
                                          target="ig_residualized_ols_lora_scgpt")


def task_attention_residualized_ols_lora_geneformer(data_dir: Path, results_dir: Path, opts=None) -> None:
    task_attention_residualized_ols_geneformer(data_dir, results_dir, opts, source="attention_lora_geneformer",
                                               target="attention_residualized_ols_lora_geneformer")


MODEL_TASKS = {
    "attention_scgpt": task_attention_scgpt,
    "ig_lora_scgpt": task_ig_lora_scgpt,
    "attention_geneformer": task_attention_geneformer,
    "attention_lora_geneformer": task_attention_lora_geneformer,
}

# CPU tasks computed from the model tasks' output
DERIVED_TASKS = {
    "attention_residualized_ols_scgpt": task_attention_residualized_ols_scgpt,
    "attention_residualized_ols_geneformer": task_attention_residualized_ols_geneformer,
    "ig_residualized_ols_lora_scgpt": task_ig_residualized_ols_lora_scgpt,
    "attention_residualized_ols_lora_geneformer": task_attention_residualized_ols_lora_geneformer,
}

POSTPROCESS_TASKS = {
    "enrichment": task_enrichment,
    "enrichment_summary": task_enrichment_summary,
    "gwas_similarity": task_gwas_similarity,
    "gene_matrix": task_gene_matrix,
    "plots": task_plots,
}

TASKS = {**MODEL_TASKS, **DERIVED_TASKS, **POSTPROCESS_TASKS}
# Post-hoc tasks that compute GO3 similarities (take --distances)
GO3_TASKS = {"gwas_similarity", "gene_matrix"}


def _gpu_worker(task_name, gpu_id, cell_types, data_dir, results_dir, opts):
    """
    One independent process per GPU.

    This is intentionally process-based rather than torch.nn.DataParallel.
    Captum IntegratedGradients performs backward passes with respect to `vals`,
    so independent model replicas are safer and simpler here.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Prevent each worker from consuming all CPU threads.
    torch.set_num_threads(1)
    np.random.seed(SEED)
    # The workers share one output: progress as lines, not redrawn bars
    progress.use_line_mode()

    print(
        f"[worker GPU {gpu_id}] starting {task_name}; "
        f"{len(cell_types)} cell types",
        flush=True,
    )
    MODEL_TASKS[task_name](data_dir, results_dir, device, cell_types, opts)
    print(f"[worker GPU {gpu_id}] finished", flush=True)


def _parse_gpus(value):
    try:
        gpus = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--gpus must be a comma-separated list, e.g. 0,1"
        ) from exc

    if not gpus:
        raise argparse.ArgumentTypeError("--gpus cannot be empty")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus contains duplicates")
    if any(g < 0 for g in gpus):
        raise argparse.ArgumentTypeError("GPU ids must be >= 0")

    return gpus


def _run_multi_gpu(task_name, gpus, data_dir, results_dir, opts):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--gpus requires CUDA, but torch.cuda.is_available() is False"
        )

    n_available = torch.cuda.device_count()
    invalid = [g for g in gpus if g >= n_available]
    if invalid:
        raise RuntimeError(
            f"Requested GPU(s) {invalid}, but PyTorch only sees "
            f"{n_available} GPU(s)."
        )

    # Used only to calculate a balanced cell-type assignment.
    counts = _cell_type_counts(task_name, data_dir, opts["max_cells_per_type"])
    assignments, loads = _split_cell_types(counts, len(gpus))

    print("\nMulti-GPU assignment:", flush=True)
    for gpu_id, cell_types, load in zip(gpus, assignments, loads):
        print(
            f"  GPU {gpu_id}: {len(cell_types)} cell types, "
            f"{load:,} cells",
            flush=True,
        )
        for ct in cell_types:
            print(f"    - {ct}", flush=True)

    # Important for CUDA: do not fork a process after CUDA has been initialized.
    ctx = get_context("spawn")
    processes = []

    for gpu_id, cell_types in zip(gpus, assignments):
        if not cell_types:
            print(f"WARNING: GPU {gpu_id} received no cell types.", flush=True)
            continue

        p = ctx.Process(
            target=_gpu_worker,
            args=(task_name, gpu_id, cell_types, data_dir, results_dir, opts),
        )
        p.start()
        processes.append((gpu_id, p))

    failed = []
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append((gpu_id, p.exitcode))

    if failed:
        details = ", ".join(
            f"GPU {gpu}: exit code {code}" for gpu, code in failed
        )
        raise RuntimeError(f"One or more GPU workers failed: {details}")


def main():
    np.random.seed(SEED)

    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS) + ["all", "postprocess"], default="all",
                   help="all: model tasks + OLS tasks; postprocess: every post-hoc task in order.")
    p.add_argument("--dataset", choices=list(DATASET_DIRS), default="A")

    # Single-GPU device for model tasks.
    p.add_argument("--device", default="cuda:0")

    # Multi-GPU option for model tasks.
    p.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=None,
        help="Comma-separated GPU ids, e.g. 0,1",
    )

    p.add_argument("--donor-col", default=None,
                   help="Donor/subject column in h5ad obs; auto-detected if omitted. "
                        "Only used with --permutation.")
    p.add_argument("--permutation", action="store_true",
                   help="Also run the donor-level permutation selector (scGPT only, slow).")
    p.add_argument("--max-cells-per-type", type=int, default=None,
                   help="Seeded random subset of at most N cells per cell type (quick test runs).")
    p.add_argument("--batch-scale", default="auto",
                   help="Multiplier of the model batch sizes (tuned for 8 GB); 'auto' (default) "
                        "scales with GPU memory: x1 per 10 GiB, up to x8.")

    p.add_argument("--results-dir", type=Path, default=_HERE / "results")
    p.add_argument("--selection", choices=SELECTIONS + ["all"], default="all",
                   help="Gene-set selection used by the post-hoc tasks (default: all).")
    p.add_argument("--distances", nargs="+", choices=GO3_DISTANCES, default=GO3_DISTANCES,
                   help="GO term similarities for gwas_similarity/gene_matrix (default: all three; "
                        "the plots use SimRel).")
    args = p.parse_args()
    if args.batch_scale != "auto" and not args.batch_scale.isdigit():
        p.error("--batch-scale must be 'auto' or a positive integer")
    selections = SELECTIONS if args.selection == "all" else [args.selection]
    opts = {"donor_col": args.donor_col, "permutation": args.permutation,
            "max_cells_per_type": args.max_cells_per_type, "batch_scale": args.batch_scale}

    # Each dataset writes to its own folder (results/dataset_a, results/dataset_b),
    # so the post-hoc analyses never mix the two
    data_dir = DATASET_DIRS[args.dataset]
    results_dir = args.results_dir / f"dataset_{args.dataset.lower()}"

    if args.gpus is not None:
        if args.task not in MODEL_TASKS:
            raise SystemExit(
                "--gpus is only supported with GPU model tasks: "
                f"{', '.join(MODEL_TASKS)}. --task all is not a multi-GPU task."
            )

        print(f"\n# {args.task} | Dataset {args.dataset} | GPUs {args.gpus} #", flush=True)
        _run_multi_gpu(args.task, args.gpus, data_dir, results_dir, opts)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --task all runs the GPU-bound model tasks and then the OLS tasks built on
    # their output; --task postprocess runs the post-hoc analyses in order.
    names = {"all": list(MODEL_TASKS) + list(DERIVED_TASKS),
             "postprocess": list(POSTPROCESS_TASKS)}.get(args.task, [args.task])
    for name in names:
        print(f"\n# {name} | Dataset {args.dataset} #", flush=True)
        if name in MODEL_TASKS:
            TASKS[name](data_dir, results_dir, device, None, opts)
        elif name in DERIVED_TASKS:
            TASKS[name](data_dir, results_dir, opts)
        elif name in GO3_TASKS:
            TASKS[name](results_dir, selections, distances=args.distances)
        else:
            TASKS[name](results_dir, selections)


if __name__ == "__main__":
    main()
