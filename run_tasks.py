#!/usr/bin/env python
"""
Interpretability outputs for the scGPT and Geneformer models:

  1. scGPT zero-shot CLS-attention signed score (PD vs Control) -> top-30 genes,
     plus the CSSI-max score over Leiden strata and the dynamic selections
     (top30 / z-score / otsu / elbow) of both.
  2. Integrated Gradients on the LoRA fine-tuned scGPT model -> top-30 genes.
  3. The same attention score as 1 with the zero-shot Geneformer model.
  4. The same attention score with the LoRA fine-tuned Geneformer model, only
     on its correctly classified cells.
  5. OLS residualization of the scGPT zero-shot attention -> all genes +
     selections.

Runs on the full dataset (train + val + test combined), on Dataset A or
Dataset B (--dataset), each with its own results folder.

GPU-bound model tasks (1-4) can use --gpus to run independent cell-type
workers on multiple GPUs. `attention_residualized_ols_scgpt` runs on the
single --device GPU: its scGPT forward pass needs CUDA (the flash-attention
backend is fp16/bf16-only), while the regression/selection stage is numpy and
is never routed through the multi-GPU model worker.
"""

import argparse
import csv
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
from pipeline.postprocess.gwas_similarity import task_gwas_similarity
from pipeline.postprocess.plots import task_plots
from pipeline.steps import _step7_common as common
from pipeline.steps import _step7_geneformer as gf
from pipeline.steps import attention_cssi_leiden as cssi
from pipeline.steps import attention_dynamic_selection as dynamic
from pipeline.steps import step7_attention_residualized_ols as residualized_ols
from pipeline.steps import step7_explainability_ig_scgpt as ig
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


def _cell_type_subsets_scgpt(adata, allowed_cell_types=None):
    """Yield (cell type, slug, AnnData subset), optionally restricted to assigned cell types."""
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    allowed = None if allowed_cell_types is None else set(allowed_cell_types)

    for ct in sorted(values.unique()):
        if allowed is not None and ct not in allowed:
            continue
        mask = (values == ct).to_numpy()
        yield ct, common._slug_cell_type(ct), adata[mask].copy()


def _cell_type_subsets_geneformer(ds, allowed_cell_types=None):
    """Same as _cell_type_subsets_scgpt but for a Hugging Face dataset."""
    values = np.array(ds[C.CELL_TYPE_COL])
    allowed = None if allowed_cell_types is None else set(allowed_cell_types)

    for ct in sorted(set(values)):
        if allowed is not None and ct not in allowed:
            continue
        yield ct, common._slug_cell_type(ct), ds.select(np.where(values == ct)[0])


def _cell_type_counts(task_name: str, data_dir: Path) -> dict:
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


def task_attention_scgpt(data_dir: Path, results_dir: Path, device: torch.device,
                         cell_types=None, donor_col=None) -> None:
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset_scgpt(data_dir)
    model = common._load_zero_shot_model(vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets_scgpt(adata, cell_types):
        print(f"\n# attention_scgpt | {ct} | {device} #", flush=True)
        pt = common._tokenise(sub, vocab)
        donor_column = dynamic.resolve_donor_column(sub, donor_col)
        donor_ids = sub.obs[donor_column].astype(str).to_numpy()
        print(f"  donor column: {donor_column} | {np.unique(donor_ids).size} donors", flush=True)

        means, cssi_scores, gate, embeddings, strata, meta = cssi.compute_attention_cssi(
            model,
            pt,
            vocab,
            model_configs["nheads"],
            device,
            n_cls=2,
            n_neighbors=CSSI_N_NEIGHBORS,
            target_k=CSSI_TARGET_K,
            min_cells_per_condition=CSSI_MIN_CELLS_PER_CONDITION,
            random_state=SEED,
        )
        print(
            f"  CSSI: {meta.get('n_clusters', 1)} Leiden strata "
            f"(resolution={meta.get('resolution', 0):.4g}), "
            f"{meta.get('n_embedding_dims', embeddings.shape[1])}-D embeddings",
            flush=True,
        )

        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}", flush=True)
            continue

        pooled_scores = common._gene_scores(means, _signed_score)
        out_dir = results_dir / "attention_zero_shot_scgpt" / slug

        # Preserve the original baseline/output consumed by downstream tasks.
        _write_top_n(pooled_scores, out_dir / "top30_genes.csv")
        _write_all_scores(cssi_scores, out_dir / "cssi_genes.csv")
        _write_all_scores(pooled_scores, out_dir / "csv" / "all_genes.csv")

        donor_data = dynamic.collect_donor_attention_data(
            model,
            pt,
            vocab,
            model_configs["nheads"],
            device,
            donor_ids,
            need_cssi=True,
            n_neighbors=CSSI_N_NEIGHBORS,
            target_k=CSSI_TARGET_K,
            random_state=SEED,
            fixed_strata=strata,
        )

        # Dynamic top-N: pooled attention and CSSI separately. Permutation is
        # donor-level and therefore uses the same donor identities in both.
        pooled_summary = dynamic.write_selection_family(
            pooled_scores,
            out_dir / "csv",
            donor_data=donor_data,
            cssi_mode=False,
            random_state=SEED,
            permutation_n=DYNAMIC_PERMUTATIONS,
            permutation_alpha=DYNAMIC_PERMUTATION_ALPHA,
            permutation_effect_size=DYNAMIC_PERMUTATION_EFFECT_SIZE,
        )
        cssi_summary = dynamic.write_selection_family(
            cssi_scores,
            out_dir / "cssi" / "csv",
            donor_data=donor_data,
            cssi_mode=True,
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
        print(
            "  pooled dynamic N: "
            + ", ".join(f"{k}={v['n_selected']}" for k, v in pooled_summary.items()),
            flush=True,
        )
        print(
            "  CSSI dynamic N: "
            + ", ".join(f"{k}={v['n_selected']}" for k, v in cssi_summary.items()),
            flush=True,
        )

    del model
    torch.cuda.empty_cache()


def task_ig_lora_scgpt(data_dir: Path, results_dir: Path, device: torch.device,
                       cell_types=None, donor_col=None) -> None:
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset_scgpt(data_dir)

    print(f"[{device}] loading LoRA model...", flush=True)
    model = common._load_lora_model(data_dir / LORA_FULL_MODEL_SCGPT, vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets_scgpt(adata, cell_types):
        print(f"\n=== ig_lora_scgpt | {ct} | {device} ===", flush=True)

        pt = common._tokenise(sub, vocab)

        mean_abs, gate = ig._compute_ig_importance(model, pt, vocab, device)

        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}", flush=True)
            continue

        _write_top_n(mean_abs, results_dir / "ig_lora_scgpt" / slug / "top30_genes.csv")

    del model
    torch.cuda.empty_cache()


# Attention per gene per condition of one Geneformer model, for every cell type
def _attention_geneformer(model, method: str, use_predictions: bool, data_dir: Path,
                          results_dir: Path, device: torch.device, cell_types=None) -> None:
    # See geneformer_model.py for this functions description
    token_dict, gene_names = load_token_dictionary(), load_gene_names(data_dir)
    ds = _load_full_dataset_geneformer(data_dir)
    for ct, slug, sub in _cell_type_subsets_geneformer(ds, cell_types):
        print(f"\n# {method} | {ct} | {device} #", flush=True)
        pt = gf._to_inputs(sub)
        means, gate = gf._compute_attn_per_gene_per_condition(
            model, pt, token_dict, gene_names, device, use_predictions,
        )
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}", flush=True)
            continue

        # Same score and top N file as the scGPT attention
        scores = common._gene_scores(means, _signed_score)
        _write_top_n(scores, results_dir / method / slug / "top30_genes.csv")


def task_attention_geneformer(data_dir: Path, results_dir: Path, device: torch.device,
                              cell_types=None, donor_col=None) -> None:
    # Like the scGPT attention task: every cell, grouped by its true label
    model = gf._load_zero_shot_model(device)
    _attention_geneformer(model, "attention_zero_shot_geneformer", False, data_dir, results_dir,
                          device, cell_types)
    del model
    torch.cuda.empty_cache()


def task_attention_lora_geneformer(data_dir: Path, results_dir: Path, device: torch.device,
                                   cell_types=None, donor_col=None) -> None:
    # With the LoRA classifier only the correctly classified cells are used
    model = gf._load_lora_model(data_dir / LORA_DIR_GENEFORMER, device)
    _attention_geneformer(model, "attention_lora_geneformer", True, data_dir, results_dir,
                          device, cell_types)
    del model
    torch.cuda.empty_cache()


def task_attention_residualized_ols_scgpt(data_dir: Path, results_dir: Path, device: torch.device,
                                          donor_col=None) -> None:
    residualized_ols.task_attention_residualized_ols(
        _load_full_dataset_scgpt(data_dir), results_dir, device, donor_col)
    torch.cuda.empty_cache()


MODEL_TASKS = {
    "attention_scgpt": task_attention_scgpt,
    "ig_lora_scgpt": task_ig_lora_scgpt,
    "attention_geneformer": task_attention_geneformer,
    "attention_lora_geneformer": task_attention_lora_geneformer,
}

SINGLE_GPU_TASKS = {
    "attention_residualized_ols_scgpt": task_attention_residualized_ols_scgpt,
}

POSTPROCESS_TASKS = {
    "enrichment": task_enrichment,
    "enrichment_summary": task_enrichment_summary,
    "gwas_similarity": task_gwas_similarity,
    "gene_matrix": task_gene_matrix,
    "plots": task_plots,
}

TASKS = {**MODEL_TASKS, **SINGLE_GPU_TASKS, **POSTPROCESS_TASKS}


def _gpu_worker(task_name, gpu_id, cell_types, data_dir, results_dir, donor_col=None):
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

    print(
        f"[worker GPU {gpu_id}] starting {task_name}; "
        f"{len(cell_types)} cell types",
        flush=True,
    )
    MODEL_TASKS[task_name](data_dir, results_dir, device, cell_types, donor_col)
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


def _run_multi_gpu(task_name, gpus, data_dir, results_dir, donor_col=None):
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
    assignments, loads = _split_cell_types(_cell_type_counts(task_name, data_dir), len(gpus))

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
            args=(task_name, gpu_id, cell_types, data_dir, results_dir, donor_col),
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
    p.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
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
                   help="Donor/subject column in h5ad obs; auto-detected if omitted.")

    p.add_argument("--results-dir", type=Path, default=_HERE / "results")
    p.add_argument("--selection", choices=SELECTIONS + ["all"], default="all",
                   help="Gene-set selection used by the post-hoc tasks (default: all).")
    args = p.parse_args()
    selections = SELECTIONS if args.selection == "all" else [args.selection]

    # Each dataset writes to its own folder (results/dataset_a, results/dataset_b),
    # so the post-hoc analyses never mix the two
    data_dir = DATASET_DIRS[args.dataset]
    results_dir = args.results_dir / f"dataset_{args.dataset.lower()}"

    if args.gpus is not None:
        if args.task not in MODEL_TASKS:
            raise SystemExit(
                "--gpus is only supported with GPU model tasks: "
                f"{', '.join(MODEL_TASKS)}. attention_residualized_ols_scgpt uses "
                "the single --device GPU, and --task all is not a multi-GPU task."
            )

        print(f"\n# {args.task} | Dataset {args.dataset} | GPUs {args.gpus} #", flush=True)
        _run_multi_gpu(args.task, args.gpus, data_dir, results_dir, args.donor_col)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.task in SINGLE_GPU_TASKS:
        print(f"\n# {args.task} | Dataset {args.dataset} | {device} #", flush=True)
        SINGLE_GPU_TASKS[args.task](data_dir, results_dir, device, args.donor_col)
        return

    # --task all only runs the GPU-bound model tasks, because the post-hoc
    # analyses depend on their output and are run individually.
    for name in (list(MODEL_TASKS) if args.task == "all" else [args.task]):
        print(f"\n# {name} | Dataset {args.dataset} #", flush=True)
        if name in MODEL_TASKS:
            TASKS[name](data_dir, results_dir, device)
        else:
            TASKS[name](results_dir, selections)


if __name__ == "__main__":
    main()
