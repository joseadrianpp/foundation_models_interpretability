#!/usr/bin/env python
"""Robustness of the scGPT IG to dropout (Quantus' robustness family: Avg- / Max-Sensitivity).

scRNA-seq misses part of the genes a cell expresses (dropout). The
perturbation here is that: --drop (default 10 %) of the cell's expressed
genes, drawn at random, are removed from its tokens (as if not detected; the
other genes keep their binned values). The IG of the perturbed cell (same
output: PD logit towards the true class, or SIGnature's for an encoder) is
compared with the pipeline's IG of the real cell over the genes both keep,
each scaled to unit L1 norm there (the comparison is of how the attribution
is spread, not of its total):

  avg_sens / max_sens   ||a - a'||_2 / ||a||_2, mean / max over the --samples
                        perturbations (quantus.AvgSensitivity / MaxSensitivity
                        with this perturbation); 0 = identical, ~1.4 = unrelated
  spearman              rank correlation of a and a'
  top30                 Jaccard of their 30 genes with the largest attribution
  out_change            how much the model's output moved: |p - p'| of the
                        true-class probability (classifiers), 1 - cosine of
                        the cell embeddings (encoders)

An explanation that changes a lot when the output barely moves is fragile.
Cells: --cells-per-type of the FG evaluation's cells (attribution_eval/cells.csv).

Outputs in <results-dir>/dataset_<x>/robustness/: cells.csv and summary.csv.

  CUDA_VISIBLE_DEVICES=1 python evaluate_robustness.py --dataset A B --results-dir results/scgpt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import evaluate_attributions as ea
import run_tasks as rt
from pipeline import config as C
from pipeline.steps import step7_explainability_ig_scgpt as ig
from pipeline.utils.progress import progress
from pipeline.utils.scgpt_model import load_model_configs, load_vocab

SEED = 42
TOP_K = 30


def _unit(a: np.ndarray) -> np.ndarray:
    return a / max(np.abs(a).sum(), 1e-12)


def _output(net, gids: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor, classifier: bool) -> np.ndarray:
    """PD logit (classifier) or cell embedding (encoder) of a batch."""
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=gids.device.type == "cuda"):
        out = ig._pd_logit(net, gids, vals, mask) if classifier else ig._cell_embedding(net, gids, vals, mask)
    return out.float().cpu().numpy()


def run(dataset: str, results_root: Path, device: torch.device, n_samples: int, drop: float,
        cells_per_type: int) -> None:
    results_dir = results_root / f"dataset_{dataset.lower()}"
    data_dir = rt.DATASET_DIRS[dataset]
    vocab, model_configs = load_vocab(), load_model_configs()
    pad_id = vocab[C.PAD_TOKEN]
    cells = pd.read_csv(results_dir / "attribution_eval" / "cells.csv")
    cells = cells.groupby("cell_type", group_keys=False).head(cells_per_type)
    adata = rt._load_full_dataset_scgpt(data_dir)
    tokenised = {slug: pt for _, slug, pt, _, _ in ea._scgpt_cell_types(adata, vocab, None)}

    rows = []
    for form in rt.FORMS:
        classifier = form in rt.CLASSIFIER_FORMS
        output = "pd_logit" if classifier else "signature"
        net = rt._load_scgpt(form, data_dir, vocab, model_configs, device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        for slug, sel in cells.groupby("cell_type"):
            pt = tokenised[slug]
            trained = ea._PipelineIG(results_dir / f"ig_{form}_scgpt" / slug / "cell_scores.npz")
            for r in progress(list(sel.itertuples(index=False)), desc=f"robustness {dataset} {form} {slug}",
                              unit="cell"):
                i = int(r.cell)
                gids_all = pt["gene_ids"][i]
                keep = (gids_all != pad_id).numpy()
                tokens = gids_all.numpy()[keep]
                values = pt["values"][i].numpy()[keep].astype(np.float32)
                sign = 1.0 if (not classifier or r.label == 1) else -1.0
                a = trained.cell(i, tokens)[1:] * sign
                n_genes = tokens.size - 1
                n_keep = n_genes - max(1, int(round(drop * n_genes)))
                # n_samples perturbed copies: <cls> plus a random n_keep of the genes, in their order
                rng = np.random.default_rng([SEED, i, rt.FORMS.index(form)])
                picks = [np.sort(rng.choice(n_genes, n_keep, replace=False)) for _ in range(n_samples)]
                pos = np.stack([np.concatenate([[0], p + 1]) for p in picks])
                gids = torch.as_tensor(tokens[pos], device=device)
                vals = torch.as_tensor(values[pos], device=device)
                vals[:, 0] = 0.0
                mask = gids.eq(pad_id)
                a_pert = ig._ig_batch(net, gids, vals, mask, device, ig.IG_STEPS, [ig.IG_INTERNAL_BATCH_SIZE],
                                      output)[:, 1:] * sign
                # Output of the real cell and of the perturbed ones
                real = _output(net, torch.as_tensor(tokens[None], device=device),
                               torch.as_tensor(values[None], device=device),
                               torch.zeros((1, tokens.size), dtype=torch.bool, device=device), classifier)
                pert = _output(net, gids, vals, mask, classifier)
                if classifier:
                    prob = lambda z: 1.0 / (1.0 + np.exp(-sign * z))
                    out_change = np.abs(prob(real[0]) - prob(pert))
                else:
                    cos = (pert @ real[0]) / (np.linalg.norm(pert, axis=1) * np.linalg.norm(real[0]))
                    out_change = 1.0 - cos
                sens, rho, jac = [], [], []
                for k, p in enumerate(picks):
                    u, v = _unit(a[p]), _unit(a_pert[k])
                    sens.append(np.linalg.norm(u - v) / max(np.linalg.norm(u), 1e-12))
                    rho.append(spearmanr(u, v)[0])
                    top_u, top_v = set(np.argsort(-u)[:TOP_K]), set(np.argsort(-v)[:TOP_K])
                    jac.append(len(top_u & top_v) / len(top_u | top_v))
                rows.append({"form": form, "cell_type": slug, "cell": i, "label": r.label, "n_genes": n_genes,
                             "avg_sens": np.mean(sens), "max_sens": np.max(sens), "spearman": np.mean(rho),
                             "top30": np.mean(jac), "out_change": float(np.mean(out_change))})
        del net
        torch.cuda.empty_cache()
        print(f"  {dataset} {form}: done", flush=True)

    out_dir = results_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cells.csv", index=False)
    summary = df.groupby("form")[["avg_sens", "max_sens", "spearman", "top30", "out_change"]].mean()
    summary.to_csv(out_dir / "summary.csv")
    print(f"\n# {dataset}: IG under {drop:.0%} dropout, {n_samples} perturbations per cell (mean over cells)\n"
          f"{summary.round(3).to_string()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--results-dir", type=Path, default=Path("results/scgpt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=10, help="Perturbations per cell.")
    p.add_argument("--drop", type=float, default=0.1, help="Fraction of the expressed genes removed.")
    p.add_argument("--cells-per-type", type=int, default=100)
    args = p.parse_args()
    for d in args.dataset:
        run(d, args.results_dir, torch.device(args.device), args.samples, args.drop, args.cells_per_type)


if __name__ == "__main__":
    main()
