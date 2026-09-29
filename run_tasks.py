#!/usr/bin/env python
"""
Interpretability outputs for the scGPT and Geneformer models:

  1. Zero-shot CLS-attention signed score (PD vs Control) -> top-30 genes.
  2. Integrated Gradients on the LoRA fine-tuned scGPT model -> top-30 genes.
  3. The same attention score as 1 with the zero-shot Geneformer model.
  4. The same attention score with the LoRA fine-tuned Geneformer model, only
     on its correctly classified cells.

Runs on the full dataset (train + val + test combined), on Dataset A or
Dataset B (--dataset), each with its own results folder.

Run separately for each cell type.
"""

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from datasets import concatenate_datasets, load_from_disk

from pipeline import config as C
from pipeline.postprocess.enrichment import task_enrichment
from pipeline.postprocess.enrichment_summary import task_enrichment_summary
from pipeline.postprocess.gene_matrix import task_gene_matrix
from pipeline.postprocess.gwas_similarity import task_gwas_similarity
from pipeline.steps import _step7_common as common
from pipeline.steps import _step7_geneformer as gf
from pipeline.steps import step7_explainability_ig_scgpt as ig
from pipeline.utils.geneformer_model import load_gene_names, load_token_dictionary
from pipeline.utils.scgpt_model import load_model_configs, load_vocab

DATA_DIR = _HERE / "data"
# The two datasets have the same four cell types and the same kind of split,
# and each folder holds its own cells and LoRA models
DATASET_DIRS = {"A": DATA_DIR / "dataset_a", "B": DATA_DIR / "dataset_b"}

# scGPT files inside a dataset folder
TRAIN_H5AD_SCGPT = Path("mixed_split_scgpt") / "Mixed_HVG_1000_pc_noXY_noMT_mixed_train.h5ad"
VAL_H5AD_SCGPT = Path("mixed_split_scgpt") / "Mixed_HVG_1000_pc_noXY_noMT_mixed_val.h5ad"
TEST_H5AD_SCGPT = Path("mixed_split_scgpt") / "Mixed_HVG_1000_pc_noXY_noMT_mixed_test.h5ad"
LORA_FULL_MODEL_SCGPT = Path("step5_1_lora_scgpt") / "best_full_model.pt"

# Same cells and splits for Geneformer, already tokenized (Hugging Face datasets)
TRAIN_DATASET_GENEFORMER = Path("mixed_split_geneformer") / "Mixed_pc_noXY_noMT_mixed_train.dataset"
VAL_DATASET_GENEFORMER = Path("mixed_split_geneformer") / "Mixed_pc_noXY_noMT_mixed_val.dataset"
TEST_DATASET_GENEFORMER = Path("mixed_split_geneformer") / "Mixed_pc_noXY_noMT_mixed_test.dataset"
LORA_DIR_GENEFORMER = Path("step5_1_lora_geneformer")

TOP_N = 30

# scGPT's binning spreads tied expression values with the global numpy RNG
# (scgpt.preprocess._digitize), so without this the top-30 genes change on
# every run. Record it alongside the results: it fixes one draw, it does not
# make the ranking independent of the draw.
SEED = 42

# This function is pretty simple, but is because previously I had another type of score
def _signed_score(mean_pd: float, mean_ctrl: float) -> float:
    return mean_pd - mean_ctrl

# Write the csv with the top N genes 
def _write_top_n(score_map: dict, out_csv: Path, n: int = TOP_N) -> None:
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene", "score"])
        for r, (g, s) in enumerate(ranked[:n], start=1):
            w.writerow([r, g, s])

def _load_full_dataset_scgpt(data_dir: Path) -> ad.AnnData:
    train = sc.read_h5ad(data_dir / TRAIN_H5AD_SCGPT)
    val = sc.read_h5ad(data_dir / VAL_H5AD_SCGPT)
    test = sc.read_h5ad(data_dir / TEST_H5AD_SCGPT)
    return ad.concat([train, val, test])

# This function is just for getting the cell-type subsets
# Return the name of the cell type (ct), a slug for the cell type (slug, corrected name)
# and the subset of the adata for that cell type (sub)
def _cell_type_subsets_scgpt(adata):
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    for ct in sorted(values.unique()):
        yield ct, common._slug_cell_type(ct), adata[(values == ct).to_numpy()].copy()

def task_attention_scgpt(data_dir: Path, results_dir: Path, device: torch.device) -> None:
    # See scgpt_model.py for this functions description
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset_scgpt(data_dir)
    model = common._load_zero_shot_model(vocab, model_configs, device)
    for ct, slug, sub in _cell_type_subsets_scgpt(adata):
        print(f"\n# attention_scgpt | {ct} #")
        # Tokenise of the vocav of our data for the current cell type
        pt = common._tokenise(sub, vocab)
        # Attention per gene per condition
        means, gate = common._compute_attn_per_gene_per_condition(
            model, pt, vocab, model_configs["nheads"], device, n_cls=2,
            use_predictions=False,
        )
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}")
            continue

        # Average of attention per gene per condition
        scores = common._gene_scores(means, _signed_score)

        # Write the csv with the top N genes
        _write_top_n(scores, results_dir / "attention_zero_shot_scgpt" / slug / "top30_genes.csv")
    del model
    torch.cuda.empty_cache()

def task_ig_lora_scgpt(data_dir: Path, results_dir: Path, device: torch.device) -> None:
    # See scgpt_model.py for this functions description
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset_scgpt(data_dir)
    model = common._load_lora_model(data_dir / LORA_FULL_MODEL_SCGPT, vocab, model_configs, device)
    for ct, slug, sub in _cell_type_subsets_scgpt(adata):
        print(f"\n=== ig_lora_scgpt | {ct} ===")
        # Tokenise of the vocav of our data for the current cell type
        pt = common._tokenise(sub, vocab)
        # IG per gene per condition
        mean_abs, gate = ig._compute_ig_importance(model, pt, vocab, device)
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}")
            continue
        
        # Write the csv with the top N genes
        _write_top_n(mean_abs, results_dir / "ig_lora_scgpt" / slug / "top30_genes.csv")
    del model
    torch.cuda.empty_cache()

# Same as _load_full_dataset_scgpt but with the tokenized Geneformer cells
def _load_full_dataset_geneformer(data_dir: Path):
    parts = [load_from_disk(str(data_dir / p)) for p in (TRAIN_DATASET_GENEFORMER, VAL_DATASET_GENEFORMER, TEST_DATASET_GENEFORMER)]
    return concatenate_datasets(parts)

# Same as _cell_type_subsets_scgpt but for a Hugging Face dataset
def _cell_type_subsets_geneformer(ds):
    values = np.array(ds[C.CELL_TYPE_COL])
    for ct in sorted(set(values)):
        yield ct, common._slug_cell_type(ct), ds.select(np.where(values == ct)[0])

# Attention per gene per condition of one Geneformer model, for every cell type
def _attention_geneformer(model, method: str, use_predictions: bool, data_dir: Path,
                          results_dir: Path, device: torch.device) -> None:
    # See geneformer_model.py for this functions description
    token_dict, gene_names = load_token_dictionary(), load_gene_names(data_dir)
    ds = _load_full_dataset_geneformer(data_dir)
    for ct, slug, sub in _cell_type_subsets_geneformer(ds):
        print(f"\n# {method} | {ct} #")
        pt = gf._to_inputs(sub)
        means, gate = gf._compute_attn_per_gene_per_condition(
            model, pt, token_dict, gene_names, device, use_predictions,
        )
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}")
            continue

        # Same score and top N file as the scGPT attention
        scores = common._gene_scores(means, _signed_score)
        _write_top_n(scores, results_dir / method / slug / "top30_genes.csv")

def task_attention_geneformer(data_dir: Path, results_dir: Path, device: torch.device) -> None:
    # Like the scGPT attention task: every cell, grouped by its true label
    model = gf._load_zero_shot_model(device)
    _attention_geneformer(model, "attention_zero_shot_geneformer", False, data_dir, results_dir, device)
    del model
    torch.cuda.empty_cache()

def task_attention_lora_geneformer(data_dir: Path, results_dir: Path, device: torch.device) -> None:
    # With the LoRA classifier only the correctly classified cells are used
    model = gf._load_lora_model(data_dir / LORA_DIR_GENEFORMER, device)
    _attention_geneformer(model, "attention_lora_geneformer", True, data_dir, results_dir, device)
    del model
    torch.cuda.empty_cache()

# Tasks of the pipeline
MODEL_TASKS = {
    "attention_scgpt": task_attention_scgpt,
    "ig_lora_scgpt": task_ig_lora_scgpt,
    "attention_geneformer": task_attention_geneformer,
    "attention_lora_geneformer": task_attention_lora_geneformer,
}
POSTPROCESS_TASKS = {
    "enrichment": task_enrichment,
    "enrichment_summary": task_enrichment_summary,
    "gwas_similarity": task_gwas_similarity,
    "gene_matrix": task_gene_matrix,
}
TASKS = {**MODEL_TASKS, **POSTPROCESS_TASKS}

def main():
    # This seed is for the binning of tied expression values in scGPT's preprocess
    np.random.seed(SEED)

    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    p.add_argument("--dataset", choices=list(DATASET_DIRS), default="A")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", type=Path, default=_HERE / "results")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Each dataset writes to its own folder (results/dataset_a, results/dataset_b),
    # so the post-hoc analyses never mix the two
    data_dir = DATASET_DIRS[args.dataset]
    results_dir = args.results_dir / f"dataset_{args.dataset.lower()}"

    # --task all only runs the GPU-bound model tasks, because the post-hoc
    # analyses depend on their output and are run individually.
    for name in (list(MODEL_TASKS) if args.task == "all" else [args.task]):
        print(f"\n# {name} | Dataset {args.dataset} #")
        if name in MODEL_TASKS:
            TASKS[name](data_dir, results_dir, device)
        else:
            TASKS[name](results_dir)

if __name__ == "__main__":
    main()
