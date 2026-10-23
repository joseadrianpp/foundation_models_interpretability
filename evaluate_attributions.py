#!/usr/bin/env python
"""Evaluate per-cell gene attributions and population rankings with Quantus (Problems A and B).

scGPT (the study): the faithfulness gain (FG) of every form (zero_shot,
step2_5, mlp, lora; --forms) on the same cells, from a finished pipeline
run (run_tasks.py --task scgpt, read from --results-dir):

  cells    --cells-per-type (default 200, 800 per dataset) per cell type,
           the same for every form: cells classified correctly by both
           classifiers (zero-shot + MLP and LoRA, the `correct` their IG
           tasks saved), spread over the donors (each donor's cells in a
           seeded random order, the donors taken in turn; a donor out of
           cells leaves its turns to the others). attribution_eval/cells.csv
  methods  ig: the pipeline's signed IG of the cell (cell_scores.npz, the
           cell type tokenised exactly as the pipeline did), towards the true
           class for a classifier (IG of the PD logit times +1/2 or -1/2),
           as saved for an encoder (SIGnature); random: uniform noise;
           rank_<score>_<ranking>: every population ranking of the form
           (score ig or igsigned, ranking pooled, cssi or ols) as every
           cell's attribution. IG± is flipped for control cells only for an
           encoder (its PD - control mean of the SIGnature IG points to PD);
           for a classifier it is already the mean IG towards the true class
           of both conditions, like the cell's own ig. rollout and
           gradrollout: the cell's attention rollout and gradient-weighted
           rollout (rollout_ / gradrollout_<form>_scgpt, when those tasks ran;
           the same for both targets), and their rankings like the IG ones
           (rollout± and gradrollout± flipped for control cells in every
           form, as PD - control contrasts of non-negative scores). --add-missing evaluates only the methods a cell type's
           file lacks (random's AUC of the same cell is reused). |IG| is not
           evaluated: its FG was far lower than the signed IG's on every model
           (0.19 / 0.17 / 0.03 against 0.93 / 0.78 / 0.66, scCello /
           Geneformer / scGPT LoRA, dataset A), as it adds up genes that push
           against the target.
  curve    the MoRF (deletion) curve of quantus.PixelFlipping: genes removed
           most relevant first (value 0, the IG baseline), the output after
           each step; --engine batch computes the same steps in batches
           (_morf_curves), --engine quantus calls PixelFlipping (to validate).
           The classifiers' output is the target class probability; an
           encoder without a head has none, so its output is the one its IG
           explains, SIGnature's y = sum_k |w_k * f_k(x')| (w = f(x)), as
           (y(x') - y(0)) / (y(x) - y(0)): 1 for the real cell, 0 with every
           gene removed.
  FG       1 - deletion AUC / deletion AUC of random on the same cell (0 =
           random, higher = better; above 1 is possible for the encoders,
           whose output can fall below that of the baseline).

Outputs per form (<results-dir>/dataset_<x>/attribution_eval/scgpt_<form>/):
cells/<cell type>.csv (one row per cell and method) and cells/<cell
type>_curves.npz, written cell type by cell type (--skip-done keeps them),
then curves.csv (all of them) and fg_by_cell_type.csv (mean FG per method
and cell type, and over every cell; read by the macrotable).

  CUDA_VISIBLE_DEVICES=1 python evaluate_attributions.py --dataset A --results-dir results/scgpt --skip-done

Geneformer and scCello (earlier evaluation, one classifier, --form mlp|lora,
its own correctly classified cells): the explanations of rank_model.py
(ig, attention, random), each scored by the Quantus metrics below, or with
--curves by their MoRF and LeRF curves and the scores derived from them.

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
metric with the attribution negated), and the scores derived from them:

  deletion_auc / lerf_auc  area under each curve (output over the fraction
                           of genes removed)
  fg                       faithfulness gain, 1 - deletion_auc / deletion_auc
                           of random on the same cell (0 = random, higher =
                           better): the single Problem A / B metric
  abpc                     lerf_auc - deletion_auc, the area between the
                           curves (Samek et al., 2017): necessity and
                           sufficiency together (higher = better)
  comprehensiveness_10     p(0) - p after removing the top 10% (higher = better)
  sufficiency_10           p(0) - p keeping only the top 10%, i.e. after
                           removing the least relevant 90% (lower = better)
  k_flip                   fraction of genes removed, most relevant first,
                           until the output drops below 0.5 (lower = the
                           decision rests on fewer genes; 1 if it never does)

Outputs (<results-dir>/dataset_<x>/attribution_eval/<model>_<form>/):
cells.csv, one row per (cell, method) with every score and the seconds each
metric took, and summary.csv, the mean and standard error per method; with
--curves, curves.csv, curves_summary.csv, fg_by_cell_type.csv (mean FG per
method and cell type, and over all cells) and curves.npz (every curve, keyed
<cell type>:<cell>:<method>:<morf|lerf>).

  CUDA_VISIBLE_DEVICES=1 python evaluate_attributions.py --model scgpt --form lora --dataset A \
      --curves --rankings --results-dir results/scgpt
"""

import argparse
import gc
import math
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
from pipeline.evaluation.scgpt_model import ScGPTEmbeddingScorer, ScGPTValueClassifier
from pipeline.postprocess import gene_sets as gs
from pipeline.steps import _step7_common as common
from pipeline.steps import _step7_geneformer as gf
from pipeline.steps import attention_dynamic_selection as dynamic
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
# Default cells per cell type (spread over the donors)
CELLS_PER_TYPE = 200


def _features_in_step(n: int, steps: int = PF_STEPS) -> int:
    """The divisor of n closest to n / steps (Quantus needs n % features_in_step == 0)."""
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return min(divisors, key=lambda d: (abs(d - n / steps), d))


def _eligible(model, cell: dict) -> bool:
    """Whether the cell can be evaluated: a classifier must predict its label, an encoder takes any cell."""
    model.set_cell(cell["tokens"], cell["x"])
    if not model.softmax:
        return True
    with torch.no_grad():
        return int(model(torch.as_tensor(cell["x"]))[0].argmax()) == cell["label"]


def _donor_order(idx: np.ndarray, donors, rng: np.random.Generator) -> np.ndarray:
    """`idx` in the order cells are tried: donors in turn, each donor's cells in a seeded random order.

    Without donors (None), a seeded random order of the cells.
    """
    if donors is None:
        return rng.permutation(idx)
    donors = np.asarray(donors)
    names = rng.permutation(np.unique(donors))
    queues = [rng.permutation(idx[donors == d]) for d in names]
    rounds = max(len(q) for q in queues)
    return np.array([q[r] for r in range(rounds) for q in queues if r < len(q)], dtype=idx.dtype)


def _pick(model, candidates, n: int):
    """The first n eligible cells of `candidates` (dicts, in order)."""
    cells = []
    for cell in candidates:
        if _eligible(model, cell):
            cells.append(cell)
            if len(cells) == n:
                break
    return cells


def _rank_cells(model: RankTokenClassifier, spec: dict, data_dir, n: int, rng: np.random.Generator):
    """(cell type, slug, up to n eligible cells) of a Geneformer or scCello dataset (no donor column)."""
    ds = rt._load_full_dataset_rank(data_dir, spec)
    for ct, slug, sub in rt._cell_type_subsets_geneformer(ds):
        pt = gf._to_inputs(sub)
        order = _donor_order(np.arange(len(pt["input_ids"])), None, rng)
        candidates = ({"idx": int(i), "donor": "", "label": int(pt["condition_labels"][i]),
                       "tokens": pt["input_ids"][i], "x": pt["input_ids"][i].astype(np.float32)[None, None, :]}
                      for i in order)
        yield ct, slug, _pick(model, candidates, n)


# ---------------------------------------------------------------------------
# scGPT: FG of the four forms on the same cells, from the pipeline's IG
# ---------------------------------------------------------------------------
def _scgpt_cell_types(adata, vocab, max_cells):
    """(cell type, slug, tokens, donors) of every cell type, tokenised exactly as the pipeline did.

    The whole cell type (or the pipeline's --max-cells-per-type subset) goes
    through common._tokenise like in run_tasks.py, so the binned values and
    the kept genes, and therefore the IG of cell_scores.npz, are the same.
    """
    donor_col = dynamic.resolve_donor_column(adata, None)
    for ct, slug, sub in rt._cell_type_subsets_scgpt(adata, None, max_cells):
        pt = common._tokenise(sub, vocab)
        yield ct, slug, pt, sub.obs[donor_col].astype(str).to_numpy(), sub.obs_names.to_numpy()


def _shared_cells(results_dir: Path, slug: str, donors: np.ndarray, labels: np.ndarray, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Positions (in the cell type) of the n cells every form is evaluated on.

    Cells classified correctly by both classifiers (the `correct` the IG tasks
    of the zero-shot + MLP and LoRA forms saved), spread over the donors
    (_donor_order).
    """
    correct = np.ones(labels.size, dtype=bool)
    for form in rt.CLASSIFIER_FORMS:
        path = results_dir / f"ig_{form}_scgpt" / slug / "cell_scores.npz"
        with np.load(path) as z:
            if z["correct"].size != labels.size:
                raise SystemExit(f"{path} has {z['correct'].size} cells, the cell type {labels.size}: "
                                 "use the --max-cells-per-type of that run")
            correct &= z["correct"]
    order = _donor_order(np.arange(labels.size), donors, rng)
    return order[correct[order]][:n]


class _PipelineIG:
    """The signed IG the pipeline saved for one form and cell type, cell by cell."""

    def __init__(self, path: Path):
        with np.load(path) as z:
            self.cells, self.gids, self.ig = z["token_cells"], z["token_gids"], z["token_ig"]
        self.ptr = np.searchsorted(self.cells, np.arange(self.cells.max() + 2))

    def cell(self, i: int, gids: np.ndarray) -> np.ndarray:
        """IG of every position of cell i (<cls> first, 0): `gids` must be its gene tokens."""
        lo, hi = self.ptr[i], self.ptr[i + 1]
        if not np.array_equal(self.gids[lo:hi], gids[1:]):
            raise ValueError(f"cell {i}: the pipeline's IG tokens differ from the tokenised cell")
        return np.concatenate([[0.0], self.ig[lo:hi]]).astype(np.float32)


def _morf_curves(model, x: np.ndarray, target: int, attributions: dict, chunk: int) -> dict:
    """The MoRF (deletion) curve of every attribution, computed in batches.

    The same procedure as quantus.PixelFlipping (normalise=False,
    perturb_baseline = model.remove_value): genes ordered by np.argsort(-a),
    ceil(n / features_in_step) cumulative steps, each setting the next
    features_in_step genes to the baseline, the output read after each step;
    the output before any removal goes first. Quantus runs the steps one
    forward at a time; here every step of every attribution is one input of a
    batch of `chunk`.
    """
    n = x.shape[-1]
    f = _features_in_step(n)
    steps = math.ceil(n / f)
    flat = x.reshape(-1)
    inputs = [flat[None, :]]
    for a in attributions.values():
        order = np.argsort(-a.reshape(1, -1), axis=1)[0]
        removed = np.zeros((steps, n), dtype=bool)
        for k in range(steps):
            removed[k:, order[k * f:(k + 1) * f]] = True
        inputs.append(np.where(removed, np.float32(model.remove_value), flat[None, :]))
    batch = np.concatenate(inputs).astype(np.float32)
    outs = []
    with torch.no_grad():
        for start in range(0, len(batch), chunk):
            out = model(torch.as_tensor(batch[start:start + chunk, None, :]))
            out = torch.softmax(out, dim=1) if model.softmax else out
            outs.append(out[:, target].float().cpu().numpy())
    outs = np.concatenate(outs)
    p0, rest = outs[0], outs[1:].reshape(len(attributions), steps)
    return {m: np.concatenate([[p0], rest[j]]) for j, m in enumerate(attributions)}


def _quantus_morf(model, x: np.ndarray, target: int, attributions: dict, device) -> dict:
    """The same curves with quantus.PixelFlipping, step by step (to validate _morf_curves)."""
    with torch.no_grad():
        out = model(torch.as_tensor(x).to(model.device))
        p0 = float((torch.softmax(out, dim=1) if model.softmax else out)[0, target])
    curves = {}
    for m, a in attributions.items():
        metric = quantus.PixelFlipping(features_in_step=_features_in_step(x.shape[-1]),
                                       perturb_baseline=model.remove_value, normalise=False,
                                       disable_warnings=True, display_progressbar=False)
        result = metric(model=model, x_batch=x, y_batch=np.array([target]), a_batch=a.copy(), device=str(device),
                        channel_first=True, softmax=model.softmax)
        curves[m] = np.concatenate([[p0], np.asarray(result[0], dtype=float)])
        _free_quantus_models()
    return curves


def _auc(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve, np.linspace(0, 1, len(curve))))


def run_scgpt(args) -> None:
    """FG of the scGPT forms: per-cell IG (the pipeline's), random and the population rankings."""
    data_dir = rt.DATASET_DIRS[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = args.results_dir / f"dataset_{args.dataset.lower()}"
    rankings_dir = args.rankings_dir or args.results_dir
    eval_dir = results_dir / "attribution_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    vocab, model_configs = load_vocab(), load_model_configs()
    names = rt._load_scgpt_gene_names(data_dir)
    models = {}
    for form in args.forms:
        net = rt._load_scgpt(form, data_dir, vocab, model_configs, device)
        cls = ScGPTValueClassifier if form in rt.CLASSIFIER_FORMS else ScGPTEmbeddingScorer
        models[form] = cls(net, vocab, model_configs["nheads"]).eval()

    rng = np.random.default_rng(SEED)
    cells_csv = eval_dir / "cells.csv"
    saved_cells = pd.read_csv(cells_csv) if cells_csv.exists() else None
    adata = rt._load_full_dataset_scgpt(data_dir)
    for ct, slug, pt, donors, obs_names in _scgpt_cell_types(adata, vocab, args.max_cells_per_type):
        labels = pt["condition_labels"].numpy()
        # The same cells for every form, chosen once per dataset (cells.csv)
        if saved_cells is not None and (saved_cells["cell_type"] == slug).any():
            sel = saved_cells[saved_cells["cell_type"] == slug]["cell"].to_numpy()
        else:
            sel = _shared_cells(results_dir, slug, donors, labels, args.cells_per_type, rng)
            rows = pd.DataFrame({"cell_type": slug, "cell": sel, "obs_name": obs_names[sel],
                                 "donor": donors[sel], "label": labels[sel]})
            saved_cells = rows if saved_cells is None else pd.concat([saved_cells, rows], ignore_index=True)
            saved_cells.to_csv(cells_csv, index=False)
        print(f"\n# scGPT {args.dataset} {ct}: {sel.size} cells from {np.unique(donors[sel]).size} donors",
              flush=True)

        for form, model in models.items():
            out_dir = eval_dir / f"scgpt_{form}"
            ct_csv = out_dir / "cells" / f"{slug}.csv"
            done = pd.read_csv(ct_csv) if ct_csv.exists() and (args.skip_done or args.add_missing) else None
            ct_csv.parent.mkdir(parents=True, exist_ok=True)
            ig_scores = _PipelineIG(results_dir / f"ig_{form}_scgpt" / slug / "cell_scores.npz")
            # Per-cell rollout and gradient-weighted rollout, when their tasks ran
            per_cell = {}
            for kind in ("rollout", "gradrollout"):
                npz = results_dir / f"{kind}_{form}_scgpt" / slug / "cell_scores.npz"
                if npz.exists():
                    per_cell[kind] = _PipelineIG(npz)
            rankings = _ranking_methods("scgpt", form, rankings_dir, args.dataset, slug, names)
            methods = ["ig", "random", *per_cell, *rankings]
            # Methods to compute: all, or (--add-missing) those the saved file lacks
            todo = set(methods)
            if done is not None:
                todo = set(methods) - set(done["method"]) if args.add_missing else set()
            if not todo:
                print(f"  {form}: done ({ct_csv}), kept", flush=True)
                continue
            if done is not None:
                print(f"  {form}: adding {sorted(todo)} to {ct_csv}", flush=True)
            rows, curves_npz = [], {}
            for i in progress(sel, desc=f"scgpt_{form} {slug}", unit="cell"):
                i = int(i)
                gids = pt["gene_ids"][i].numpy()
                keep = gids != vocab[C.PAD_TOKEN]
                tokens = gids[keep]
                x = pt["values"][i].numpy()[keep].astype(np.float32)[None, None, :]
                label = int(labels[i])
                y = np.array([label])
                model.set_cell(tokens, x)
                ig = ig_scores.cell(i, tokens)
                # Signed IG towards the true class for a classifier (as explain_ig), as saved for an encoder
                if form in rt.CLASSIFIER_FORMS:
                    ig = ig * (0.5 if label == 1 else -0.5)
                attributions = {"ig": ig[None, None, :]}
                np.random.seed(SEED + i)
                attributions["random"] = scgpt_model.explain_random(model, x, y)
                for kind, scores in per_cell.items():
                    attributions[kind] = scores.cell(i, tokens)[None, None, :]
                for method, explain in rankings.items():
                    attributions[method] = explain(model, x, y)
                attributions = {m: a for m, a in attributions.items() if m in todo}
                start = time.perf_counter()
                curves = (_quantus_morf(model, x, label, attributions, device) if args.engine == "quantus"
                          else _morf_curves(model, x, label, attributions, args.chunk))
                seconds = round((time.perf_counter() - start) / len(attributions), 3)
                for method, curve in curves.items():
                    rows.append({"cell_type": slug, "cell": i, "donor": donors[i], "label": label,
                                 "n_tokens": x.shape[-1], "method": method, "deletion_auc": _auc(curve),
                                 "curve_s": seconds})
                    curves_npz[f"{slug}:{i}:{method}"] = curve
            new = pd.DataFrame(rows)
            curves_path = out_dir / "cells" / f"{slug}_curves.npz"
            if done is not None:
                # Keep the saved methods (and random's AUC, the FG reference) and their curves
                new = pd.concat([done.drop(columns="fg"), new], ignore_index=True)
                if curves_path.exists():
                    with np.load(curves_path) as old:
                        curves_npz = {**{k: old[k] for k in old.files}, **curves_npz}
            _add_fg(new).to_csv(ct_csv, index=False)
            np.savez_compressed(curves_path, **curves_npz)

    # Per form: every cell type's rows, the mean FG per method and cell type, and over all cells
    for form in args.forms:
        out_dir = eval_dir / f"scgpt_{form}"
        parts = sorted((out_dir / "cells").glob("*.csv"))
        if not parts:
            continue
        cells_df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
        cells_df.to_csv(out_dir / "curves.csv", index=False)
        fg = cells_df.pivot_table(index="method", columns="cell_type", values="fg", aggfunc="mean")
        fg["all"] = cells_df.groupby("method")["fg"].mean()
        fg.to_csv(out_dir / "fg_by_cell_type.csv")
        print(f"\n# scGPT {form} {args.dataset}: mean FG\n{fg.round(3).to_string()}", flush=True)


def _ranking_methods(model_name: str, form: str, rankings_dir: Path, dataset: str, slug: str,
                     names: dict) -> dict:
    """{method: explain function} of the population rankings (score IG or IG±) of one form and cell type.

    Every ranking the pipeline wrote for that cell type in `rankings_dir`; a
    gene's score goes to every token named like it (`names`: token id ->
    gene symbol, as the pipeline names them). IG± (mean_pd - mean_control of
    the signed IG) is flipped for control cells only for an encoder: for a
    classifier the IG is of the PD logit, so mean_pd(IG) - mean_control(IG)
    already adds up both conditions' IG towards their true class.
    """
    tokens_of = {}
    for gid, name in names.items():
        tokens_of.setdefault(name, []).append(int(gid))
    model_form = gs.FORMS[form][0]
    results_dir = rankings_dir / f"dataset_{dataset.lower()}"
    methods = {}
    for method in gs.METHODS:
        meta = gs.METHOD_META[method]
        if (meta["model"], meta["model_form"]) != (model_name, model_form) or meta["score"] == "attention":
            continue
        path = results_dir / gs.ALL_SCORE_PATHS[method].format(ct=slug)
        if not path.exists():
            print(f"  ranking {method} missing ({path}), skipped", flush=True)
            continue
        df = pd.read_csv(path)
        scores = {gid: float(score) for gene, score in zip(df["gene"].astype(str), df["score"])
                  for gid in tokens_of.get(gene, [])}
        signed = ((meta["score"] == "igsigned" and form not in rt.CLASSIFIER_FORMS)
                  or meta["score"] in ("rolloutsigned", "gradrolloutsigned"))
        methods[f"rank_{meta['score']}_{meta['ranking']}"] = scgpt_model.ranking_explainer(scores, signed=signed)
    return methods


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
        out = model(torch.as_tensor(x).to(model.device))
        p0 = float((torch.softmax(out, dim=1) if model.softmax else out)[0, int(y[0])])
    curves = {}
    for order, attribution in (("morf", a), ("lerf", -a)):
        metric = quantus.PixelFlipping(features_in_step=_features_in_step(n_tokens),
                                       perturb_baseline=model.remove_value, normalise=False,
                                       disable_warnings=True, display_progressbar=False)
        result = metric(model=model, x_batch=x, y_batch=y, a_batch=attribution.copy(), device=str(device),
                        channel_first=True, softmax=model.softmax)
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


def _add_fg(cells_df: pd.DataFrame) -> pd.DataFrame:
    """FG of every row: 1 - deletion AUC / deletion AUC of random on the same cell."""
    random_auc = (cells_df[cells_df["method"] == "random"].set_index(["cell_type", "cell"])["deletion_auc"])
    ref = cells_df.set_index(["cell_type", "cell"]).index.map(random_auc.to_dict())
    cells_df["fg"] = 1.0 - cells_df["deletion_auc"].to_numpy() / np.asarray(ref, dtype=float)
    return cells_df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=["scgpt", *rt.RANK_MODELS], default="scgpt")
    p.add_argument("--forms", nargs="+", choices=rt.FORMS, default=list(rt.FORMS),
                   help="scGPT forms, evaluated on the same cells (default: all four).")
    p.add_argument("--form", choices=rt.CLASSIFIER_FORMS, default="lora", help="Geneformer / scCello classifier.")
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    p.add_argument("--cells-per-type", type=int, default=CELLS_PER_TYPE)
    p.add_argument("--max-cells-per-type", type=int, default=None,
                   help="scGPT: the --max-cells-per-type of the pipeline run whose IG and rankings are read.")
    p.add_argument("--engine", choices=["batch", "quantus"], default="batch",
                   help="scGPT deletion curves: batched (default) or quantus.PixelFlipping step by step.")
    p.add_argument("--chunk", type=int, default=64, help="scGPT: perturbed inputs per forward pass.")
    p.add_argument("--skip-done", action="store_true", help="scGPT: keep the (form, cell type) already written.")
    p.add_argument("--add-missing", action="store_true",
                   help="scGPT: for a (form, cell type) already written, add only the methods it lacks.")
    p.add_argument("--rankings-dir", type=Path, default=None,
                   help="Results root holding the rankings (default: --results-dir).")
    p.add_argument("--methods", nargs="+", choices=list(rank_model.EXPLAIN_FUNCS), default=["ig", "random"],
                   help="Geneformer / scCello explanations.")
    p.add_argument("--metrics", nargs="+", default=None, help="Geneformer / scCello: subset of the metric names.")
    p.add_argument("--curves", action="store_true", help="Geneformer / scCello: only the MoRF/LeRF curves.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    if args.model == "scgpt":
        run_scgpt(args)
        return

    # Geneformer and scCello: one classifier, its own correctly classified cells, every Quantus metric
    methods = list(dict.fromkeys(args.methods + ["random"]))
    data_dir = rt.DATASET_DIRS[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    spec = rt.RANK_MODELS[args.model]
    token_dict = load_token_dictionary(spec["token_dict"])
    model = RankTokenClassifier(rt._load_rank(spec, args.form, data_dir, device), token_dict, spec["emb_layer"])
    model.eval()
    cell_types = _rank_cells(model, spec, data_dir, args.cells_per_type, rng)
    tag = f"{args.model}_{args.form}"
    out_dir = args.results_dir / f"dataset_{args.dataset.lower()}" / "attribution_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, all_curves = [], {}
    for ct, slug, cells in cell_types:
        print(f"\n# {tag} {args.dataset} {ct}: {len(cells)} cells, {len(methods)} methods", flush=True)
        for cell in progress(cells, desc=f"{tag} {slug}", unit="cell"):
            i, label, x = cell["idx"], cell["label"], cell["x"]
            n_tokens = x.shape[-1]
            model.set_cell(cell["tokens"], x)
            model._target = label
            y = np.array([label])
            for method in methods:
                explain = rank_model.EXPLAIN_FUNCS[method]
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
        cells_df = _add_fg(cells_df)
        cells_df.to_csv(out_dir / "curves.csv", index=False)
        np.savez_compressed(out_dir / "curves.npz", **all_curves)
        summary = cells_df.groupby("method")[["fg", *CURVE_SCORES]].agg(["mean", "sem"])
        summary.to_csv(out_dir / "curves_summary.csv")
        print(summary.round(4).to_string(), flush=True)
        return
    scores = [c for c in cells_df.columns if c not in ("cell_type", "cell", "label", "n_tokens", "method")
              and not c.endswith("_s")]
    summary = cells_df.groupby("method")[scores].agg(["mean", "sem"])
    summary.to_csv(out_dir / "summary.csv")
    print(summary.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
