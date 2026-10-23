#!/usr/bin/env python
"""Model randomisation check of the scGPT IG (Efficient MPRT, Hedström et al. 2023).

An explanation that reflects what the model learned must change when the
model's weights are replaced by random ones. Every parameter of each form
(encoder and head) is re-initialised (`reset_parameters` of every module,
seed 42, as Quantus' randomisation does), the IG is recomputed on the cells
of the FG evaluation (attribution_eval/cells.csv) with the same output (PD
logit or SIGnature) and compared with the pipeline's IG of the trained model:

  spearman      rank correlation of |IG| over the cell's genes (trained vs
                random); near 1 = the explanation does not depend on the
                weights (it follows the input), near 0 = it does
  top30         Jaccard of the 30 genes with the largest |IG|
  empr          Efficient MPRT: entropy of the random model's |IG| over the
                trained one's, minus 1 (quantus.EfficientMPRT's complexity
                rise; > 0 = the trained explanation is more concentrated)
  expr_trained  rank correlation of the trained |IG| with the cell's binned
  expr_random   expression values (IG multiplies by the input, so some
                correlation is built in), and of the random model's

Outputs in <results-dir>/dataset_<x>/mprt/: cells.csv and summary.csv.

  CUDA_VISIBLE_DEVICES=1 python evaluate_mprt.py --dataset A --results-dir results/scgpt
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import evaluate_attributions as ea
import run_tasks as rt
from pipeline import config as C
from pipeline.utils.scgpt_model import load_model_configs, load_vocab
from pipeline.steps import step7_explainability_ig_scgpt as ig
from pipeline.utils.progress import progress

SEED = 42
TOP_K = 30
BATCH = 8


def _randomised(model: torch.nn.Module) -> torch.nn.Module:
    """A copy of the model with every parameter re-initialised (seed 42)."""
    rand = copy.deepcopy(model)
    torch.manual_seed(SEED)
    for module in rand.modules():
        if module is not rand and callable(getattr(module, "reset_parameters", None)):
            module.reset_parameters()
    return rand.eval()


def _entropy(a: np.ndarray) -> float:
    p = np.abs(a) / max(np.abs(a).sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run(dataset: str, results_root: Path, device: torch.device) -> None:
    results_dir = results_root / f"dataset_{dataset.lower()}"
    data_dir = rt.DATASET_DIRS[dataset]
    vocab, model_configs = load_vocab(), load_model_configs()
    pad_id = vocab[C.PAD_TOKEN]
    cells = pd.read_csv(results_dir / "attribution_eval" / "cells.csv")
    adata = rt._load_full_dataset_scgpt(data_dir)
    tokenised = {slug: pt for _, slug, pt, _, _ in ea._scgpt_cell_types(adata, vocab, None)}

    rows = []
    for form in rt.FORMS:
        output = "pd_logit" if form in rt.CLASSIFIER_FORMS else "signature"
        net = _randomised(rt._load_scgpt(form, data_dir, vocab, model_configs, device))
        for p in net.parameters():
            p.requires_grad_(False)
        for slug, sel in cells.groupby("cell_type"):
            pt = tokenised[slug]
            trained = ea._PipelineIG(results_dir / f"ig_{form}_scgpt" / slug / "cell_scores.npz")
            idx = sel["cell"].to_numpy()
            for start in progress(range(0, idx.size, BATCH), desc=f"mprt {dataset} {form} {slug}", unit="batch"):
                b = idx[start:start + BATCH]
                gids = pt["gene_ids"][b].to(device)
                vals = pt["values"][b].to(device).float()
                rand = ig._ig_batch(net, gids, vals, gids.eq(pad_id), device, ig.IG_STEPS, [ig.IG_INTERNAL_BATCH_SIZE],
                                    output)
                for j, i in enumerate(b):
                    keep = (gids[j] != pad_id).cpu().numpy()
                    tokens = gids[j].cpu().numpy()[keep]
                    a_t = np.abs(trained.cell(int(i), tokens)[1:])
                    a_r = np.abs(rand[j][keep][1:])
                    x = vals[j].cpu().numpy()[keep][1:]
                    top_t, top_r = set(np.argsort(-a_t)[:TOP_K]), set(np.argsort(-a_r)[:TOP_K])
                    rows.append({"form": form, "cell_type": slug, "cell": int(i),
                                 "spearman": spearmanr(a_t, a_r)[0],
                                 "top30": len(top_t & top_r) / len(top_t | top_r),
                                 "empr": _entropy(a_r) / _entropy(a_t) - 1,
                                 "expr_trained": spearmanr(a_t, x)[0], "expr_random": spearmanr(a_r, x)[0]})
        del net
        torch.cuda.empty_cache()
        print(f"  {dataset} {form}: done", flush=True)

    out_dir = results_dir / "mprt"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cells.csv", index=False)
    summary = df.groupby("form")[["spearman", "top30", "empr", "expr_trained", "expr_random"]].mean()
    summary.to_csv(out_dir / "summary.csv")
    print(f"\n# {dataset}: IG of the trained model vs the randomised one (mean over cells)\n"
          f"{summary.round(3).to_string()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--results-dir", type=Path, default=Path("results/scgpt"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for d in args.dataset:
        run(d, args.results_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
