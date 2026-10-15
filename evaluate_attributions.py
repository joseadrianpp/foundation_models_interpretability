#!/usr/bin/env python
"""Evaluate per-cell gene attributions of a LoRA classifier with Quantus (Problem A).

For a model (scGPT, Geneformer or scCello) and a dataset, a seeded sample of
correctly classified cells per cell type is explained by each method (IG,
|IG|, attention, random; see pipeline/evaluation/rank_model.py and
scgpt_model.py), and each explanation is scored by Quantus metrics, one cell
at a time. Removing a gene is replacing it by <mask> (Geneformer, scCello) or
setting its expression value to 0 (scGPT), the baselines of our IG:

  faithfulness  PixelFlipping: genes removed most relevant first,
                ~50 steps; the area under the target probability curve over the
                fraction removed (deletion AUC, lower = more faithful).
                FaithfulnessCorrelation: correlation between the attribution
                of random gene subsets (10% of the genes) and the drop of the
                target logit when they are removed (higher = more faithful).
  robustness    MaxSensitivity: largest relative change of the explanation when
                2% of the genes are masked at random (lower = more robust).
  randomisation EfficientMPRT: complexity of the explanation of the fully
                randomised model over that of the trained one (higher = the
                explanation depends more on what the model learnt).
  complexity    Sparseness (Gini of |attribution|) and Complexity (entropy).

With --curves, only the two perturbation curves of each explanation are
computed (both with PixelFlipping): MoRF, the genes removed most relevant
first (the deletion curve above), and LeRF, least relevant first (the same
metric with the attribution negated). Every score derived from them is a
probability or a fraction of genes, so it compares across models:

  deletion_auc / lerf_auc  area under each curve (target probability over
                           the fraction of genes removed)
  abpc                     lerf_auc - deletion_auc, the area between the
                           curves (Samek et al., 2017): necessity and
                           sufficiency together (higher = better)
  comprehensiveness_10     p(0) - p after removing the top 10% (higher = better)
  sufficiency_10           p(0) - p keeping only the top 10%, i.e. after
                           removing the least relevant 90% (lower = better)
  k_flip                   fraction of genes removed, most relevant first,
                           until the target probability drops below 0.5
                           (lower = the decision rests on fewer genes; 1 if
                           it never flips)

Outputs (results/dataset_<x>/attribution_eval/<model>/): cells.csv, one row
per (cell, method) with every score and the seconds each metric took, and
summary.csv, the mean and standard error per method; with --curves,
curves.csv, curves_summary.csv and curves.npz (every curve, keyed
<cell type>:<cell>:<method>:<morf|lerf>).

  CUDA_VISIBLE_DEVICES=1 python evaluate_attributions.py --model sccello --dataset A --cells-per-type 12
"""

import argparse
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import quantus
import torch
from quantus.helpers.model.pytorch_model import PyTorchModel

import run_tasks as rt
from pipeline import config as C
from pipeline.evaluation import rank_model, scgpt_model
from pipeline.evaluation.rank_model import RankTokenClassifier, mask_tokens
from pipeline.evaluation.scgpt_model import ScGPTValueClassifier
from pipeline.steps import _step7_common as common
from pipeline.steps import _step7_geneformer as gf
from pipeline.utils.geneformer_model import load_token_dictionary
from pipeline.utils.progress import progress
from pipeline.utils.scgpt_model import load_model_configs, load_vocab

HERE = Path(__file__).resolve().parent
SEED = 42
# Removal steps of the deletion curve, fraction of genes per FaithfulnessCorrelation
# subset, and fraction of genes masked by the robustness perturbation
PF_STEPS = 50
FC_SUBSET = 0.10
FC_RUNS = 50
SENS_FRACTION = 0.02
SENS_SAMPLES = 10


def _features_in_step(n: int, steps: int = PF_STEPS) -> int:
    """The divisor of n closest to n / steps (Quantus needs n % features_in_step == 0)."""
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return min(divisors, key=lambda d: (abs(d - n / steps), d))


def _is_correct(model, cell: dict) -> bool:
    """Whether the classifier predicts the cell's label."""
    model.set_cell(cell["tokens"])
    with torch.no_grad():
        return int(model(torch.as_tensor(cell["x"]))[0].argmax()) == cell["label"]


def _rank_cells(model: RankTokenClassifier, spec: dict, data_dir, n: int, rng: np.random.Generator):
    """(cell type, slug, up to n correctly classified cells) of a Geneformer or scCello dataset."""
    ds = rt._load_full_dataset_rank(data_dir, spec)
    for ct, slug, sub in rt._cell_type_subsets_geneformer(ds):
        pt = gf._to_inputs(sub)
        cells = []
        for i in rng.permutation(len(pt["input_ids"])):
            ids = pt["input_ids"][i]
            cell = {"idx": int(i), "label": int(pt["condition_labels"][i]), "tokens": ids,
                    "x": ids.astype(np.float32)[None, None, :]}
            if _is_correct(model, cell):
                cells.append(cell)
            if len(cells) == n:
                break
        yield ct, slug, cells


def _scgpt_cells(model: ScGPTValueClassifier, vocab, data_dir, n: int, rng: np.random.Generator):
    """(cell type, slug, up to n correctly classified cells) of a scGPT dataset.

    Only a seeded sample of candidates (3n) is tokenised: binning the whole
    cell type takes minutes and the evaluation needs a few dozen cells. Each
    cell keeps its real tokens (no padding): <cls> and its expressed genes.
    """
    adata = rt._load_full_dataset_scgpt(data_dir)
    values = adata.obs[C.CELL_TYPE_COL].astype(str).to_numpy()
    for ct in sorted(set(values)):
        idx = np.where(values == ct)[0]
        cand = rng.permutation(idx)[:3 * n]
        pt = common._tokenise(adata[np.sort(cand)].copy(), vocab)
        cells = []
        for j in range(len(cand)):
            gids = pt["gene_ids"][j].numpy()
            keep = gids != vocab[C.PAD_TOKEN]
            cell = {"idx": int(np.sort(cand)[j]), "label": int(pt["condition_labels"][j]), "tokens": gids[keep],
                    "x": pt["values"][j].numpy()[keep].astype(np.float32)[None, None, :]}
            if _is_correct(model, cell):
                cells.append(cell)
            if len(cells) == n:
                break
        yield ct, common._slug_cell_type(ct), cells


def _metrics(n_tokens: int, remove_value: float) -> dict:
    """The Quantus metrics of one cell, (metric, call kwargs) by name."""
    common = dict(disable_warnings=True, display_progressbar=False)
    mask_id = remove_value
    sens_kwargs = {"mask_id": mask_id, "fraction": SENS_FRACTION}
    return {
        "deletion_auc": (quantus.PixelFlipping(features_in_step=_features_in_step(n_tokens),
                                               perturb_baseline=mask_id, normalise=False, **common),
                         {"softmax": True}),
        "faithfulness_corr": (quantus.FaithfulnessCorrelation(nr_runs=FC_RUNS,
                                                              subset_size=max(1, int(FC_SUBSET * n_tokens)),
                                                              perturb_baseline=mask_id, normalise=False,
                                                              return_aggregate=False, **common),
                              {"softmax": False}),
        "max_sensitivity": (quantus.MaxSensitivity(nr_samples=SENS_SAMPLES, perturb_func=mask_tokens,
                                                   perturb_func_kwargs=sens_kwargs, normalise=False, **common),
                            {"softmax": False}),
        # Its complexity is an entropy: |attribution| and class probabilities
        "efficient_mprt": (quantus.EfficientMPRT(seed=SEED, abs=True, **common), {"softmax": True}),
        "sparseness": (quantus.Sparseness(**common), {}),
        "complexity": (quantus.Complexity(**common), {}),
    }


def _score(name: str, result, x: np.ndarray, model) -> float:
    """One number per cell from what the metric returned."""
    if name == "deletion_auc":
        # Target probability before any removal, then after each step, over the fraction removed
        with torch.no_grad():
            p0 = torch.softmax(model(torch.as_tensor(x).to(model.device)), dim=1)
        curve = np.concatenate([[float(p0[0, model._target])], np.asarray(result[0], dtype=float)])
        return float(np.trapezoid(curve, np.linspace(0, 1, len(curve))))
    return float(np.asarray(result, dtype=float).ravel()[0])


CURVE_SCORES = ["deletion_auc", "lerf_auc", "abpc", "comprehensiveness_10", "sufficiency_10", "k_flip"]


def _curve_scores(model, x: np.ndarray, y: np.ndarray, a: np.ndarray, device) -> tuple:
    """MoRF and LeRF curves of one explanation (PixelFlipping) and the scores derived from them."""
    n_tokens = x.shape[-1]
    with torch.no_grad():
        p0 = float(torch.softmax(model(torch.as_tensor(x).to(model.device)), dim=1)[0, int(y[0])])
    curves = {}
    for order, attribution in (("morf", a), ("lerf", -a)):
        metric = quantus.PixelFlipping(features_in_step=_features_in_step(n_tokens),
                                       perturb_baseline=model.remove_value, normalise=False,
                                       disable_warnings=True, display_progressbar=False)
        result = metric(model=model, x_batch=x, y_batch=y, a_batch=attribution.copy(), device=str(device),
                        channel_first=True, softmax=True)
        curves[order] = np.concatenate([[p0], np.asarray(result[0], dtype=float)])
        _free_quantus_models()
    t = np.linspace(0, 1, len(curves["morf"]))
    morf, lerf = curves["morf"], curves["lerf"]
    below = np.where(morf < 0.5)[0]
    if below.size == 0:
        k_flip = 1.0
    else:
        # Linear interpolation between the last point above 0.5 and the first below
        j = below[0]
        k_flip = float(t[j - 1] + (morf[j - 1] - 0.5) / (morf[j - 1] - morf[j]) * (t[j] - t[j - 1])) if j else 0.0
    scores = {"deletion_auc": float(np.trapezoid(morf, t)), "lerf_auc": float(np.trapezoid(lerf, t))}
    scores["abpc"] = scores["lerf_auc"] - scores["deletion_auc"]
    scores["comprehensiveness_10"] = p0 - float(np.interp(0.1, t, morf))
    scores["sufficiency_10"] = p0 - float(np.interp(0.9, t, lerf))
    scores["k_flip"] = k_flip
    return scores, curves


def _free_quantus_models() -> None:
    """Release the model copies Quantus keeps alive.

    PyTorchModel caches two methods with lru_cache(maxsize=None), which holds
    every model wrapper Quantus creates, so each EfficientMPRT call (a
    randomised copy of the model) stays on the GPU: +40 MiB per call with
    scCello, out of memory after ~600 calls.
    """
    PyTorchModel._get_last_softmax_layer_index.cache_clear()
    PyTorchModel._get_model_with_linear_top.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=["scgpt", *rt.RANK_MODELS], default="sccello")
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    p.add_argument("--cells-per-type", type=int, default=12)
    p.add_argument("--methods", nargs="+", choices=list(rank_model.EXPLAIN_FUNCS),
                   default=list(rank_model.EXPLAIN_FUNCS))
    p.add_argument("--metrics", nargs="+", default=None, help="Subset of the metric names (default: all).")
    p.add_argument("--curves", action="store_true",
                   help="Only the MoRF/LeRF curves and the scores derived from them (curves.csv).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    data_dir = rt.DATASET_DIRS[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    if args.model == "scgpt":
        vocab, model_configs = load_vocab(), load_model_configs()
        net = common._load_lora_model(data_dir / rt.LORA_FULL_MODEL_SCGPT, vocab, model_configs, device)
        model = ScGPTValueClassifier(net, vocab, model_configs["nheads"]).eval()
        explain_funcs = scgpt_model.EXPLAIN_FUNCS
        cell_types = _scgpt_cells(model, vocab, data_dir, args.cells_per_type, rng)
    else:
        spec = rt.RANK_MODELS[args.model]
        token_dict = load_token_dictionary(spec["token_dict"])
        model = RankTokenClassifier(rt._load_lora_rank(spec, data_dir, device), token_dict, spec["emb_layer"])
        model.eval()
        explain_funcs = rank_model.EXPLAIN_FUNCS
        cell_types = _rank_cells(model, spec, data_dir, args.cells_per_type, rng)
    out_dir = args.results_dir / f"dataset_{args.dataset.lower()}" / "attribution_eval" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, all_curves = [], {}
    for ct, slug, cells in cell_types:
        print(f"\n# {args.model} {args.dataset} {ct}: {len(cells)} correctly classified cells", flush=True)
        for cell in progress(cells, desc=f"{args.model} {slug}", unit="cell"):
            i, label, x = cell["idx"], cell["label"], cell["x"]
            n_tokens = x.shape[-1]
            model.set_cell(cell["tokens"])
            model._target = label
            y = np.array([label])
            for method in args.methods:
                explain = explain_funcs[method]
                np.random.seed(SEED + i)
                torch.manual_seed(SEED + i)
                a = explain(model, x, y)
                row = {"cell_type": slug, "cell": i, "label": label, "n_tokens": n_tokens, "method": method}
                if args.curves:
                    start = time.perf_counter()
                    scores, curves = _curve_scores(model, x, y, a, device)
                    row.update(scores, curves_s=round(time.perf_counter() - start, 3))
                    for order, curve in curves.items():
                        all_curves[f"{slug}:{i}:{method}:{order}"] = curve
                    rows.append(row)
                    continue
                for name, (metric, call_kwargs) in _metrics(n_tokens, model.remove_value).items():
                    if args.metrics and name not in args.metrics:
                        continue
                    np.random.seed(SEED + i)
                    start = time.perf_counter()
                    result = metric(model=model, x_batch=x, y_batch=y, a_batch=a.copy(), explain_func=explain,
                                    device=str(device), channel_first=True, **call_kwargs)
                    row[name] = _score(name, result, x, model)
                    row[f"{name}_s"] = round(time.perf_counter() - start, 3)
                    _free_quantus_models()
                rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / ("curves.csv" if args.curves else "cells.csv"), index=False)

    cells_df = pd.DataFrame(rows)
    if args.curves:
        np.savez_compressed(out_dir / "curves.npz", **all_curves)
        summary = cells_df.groupby("method")[CURVE_SCORES].agg(["mean", "sem"])
        summary.to_csv(out_dir / "curves_summary.csv")
        print(summary.round(4).to_string(), flush=True)
        return
    scores = [c for c in cells_df.columns if c not in ("cell_type", "cell", "label", "n_tokens", "method")
              and not c.endswith("_s")]
    summary = cells_df.groupby("method")[scores].agg(["mean", "sem"])
    summary.to_csv(out_dir / "summary.csv")
    print(summary.round(4).to_string(), flush=True)
    print(f"\nseconds per cell and metric:\n{cells_df.groupby('method')[[f'{s}_s' for s in scores]].mean().round(2)}",
          flush=True)


if __name__ == "__main__":
    main()
