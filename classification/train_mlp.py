#!/usr/bin/env python
"""
MLP head that classifies PD vs control astrocytes from the frozen zero-shot
embeddings of one model (embed_<model>.py writes them).

The head and its training are those of pipeline_unified's Step 4, the head the
LoRA fine-tuning starts from:

  1. Training cells balanced to 50 % PD and 50 % control by dropping cells of
     the larger class at random, so the head cannot win by always answering
     the majority class. Validation and test keep all their cells.
  2. Two hidden layers of 64 units (GELU, dropout 0.3), trained with AdamW
     on the binary cross-entropy of the PD logit.
  3. Stops when the validation balanced accuracy has stopped improving and
     keeps the epoch with the lowest validation loss.
  4. Scores the test donors, who were never seen in training: per cell
     (balanced accuracy with its 95% CI, sensitivity, specificity, ROC AUC)
     and per donor (mean PD probability of its cells).

Writes in results/dataset_<x>/<model>/mlp/seed_<seed>/: best_mlp.pt,
training_history.csv, predictions.csv (every cell), test_metrics.json and
test_donors.csv.

Run in scgpt_env (it only needs torch, scikit-learn and anndata):

    python classification/train_mlp.py --dataset A --model scgpt [--seed 42]
"""

import argparse
import json
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)
from torch import nn

import config as C

_ACT_FNS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


# The same MLP as pipeline_unified (utils/mlp.py) and as the heads in ../data,
# so a head trained here loads in the interpretability code and the other way round
class FlexMLP(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, hidden_size: int,
                 dropout: float, activation: str, mlp_type: str = "constant",
                 n_classes: int = 1):
        super().__init__()
        act_cls = _ACT_FNS[activation]
        layers: List[nn.Module] = []
        in_dim, width = input_dim, hidden_size
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, width), act_cls(), nn.Dropout(dropout)]
            in_dim = width
            # Halving the width at every layer in case of "descendent"
            if mlp_type == "descendent":
                width = max(width // 2, 1)
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Drop cells of the larger diagnosis at random until both have the same number
def balance(y: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = np.bincount(y[idx]).min()
    kept = np.concatenate([rng.choice(idx[y[idx] == c], size=n, replace=False) for c in (0, 1)])
    rng.shuffle(kept)
    return kept


def predict_logits(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).squeeze(1).cpu().numpy()


# Balanced accuracy is the mean of sensitivity (PD cells called PD) and
# specificity (control cells called control), so a set with many more PD than
# control cells does not inflate it. Its 95% CI is the normal approximation
# over the cells, the one of the paper's Table S2
def cell_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    half = 1.96 * 0.5 * np.sqrt(sens * (1 - sens) / (tp + fn) + spec * (1 - spec) / (tn + fp))
    ba = balanced_accuracy_score(y, pred)
    return {
        "n_cells_pd": int(tp + fn), "n_cells_control": int(tn + fp),
        "balanced_accuracy": float(ba), "balanced_accuracy_ci95": [float(ba - half), float(ba + half)],
        "sensitivity": float(sens), "specificity": float(spec),
        "roc_auc": float(roc_auc_score(y, prob)), "f1_macro": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
          device: torch.device, seed: int):
    p = C.MLP_PARAMS
    torch.manual_seed(seed)
    model = FlexMLP(X.shape[1], p["n_layers"], p["hidden_size"], p["dropout"],
                    p["activation"], p["mlp_type"]).to(device)
    # eps=1e-4 instead of AdamW's default, as in pipeline_unified
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"], eps=1e-4)
    # One warm-up epoch from a tenth of the learning rate, then x0.9 every epoch
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1),
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)], milestones=[1])
    criterion = nn.BCEWithLogitsLoss()

    Xtr, ytr = torch.from_numpy(X[train_idx]).to(device), torch.from_numpy(y[train_idx]).float().to(device)
    rng = np.random.default_rng(seed)
    best_ba, best_loss, waited, best_state, history = -1.0, np.inf, 0, None, []
    for epoch in range(1, C.MLP_EPOCHS + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(train_idx))).to(device)
        total = 0.0
        for start in range(0, len(order), p["batch_size"]):
            batch = order[start:start + p["batch_size"]]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[batch]).squeeze(1), ytr[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch)
        scheduler.step()

        prob = 1 / (1 + np.exp(-predict_logits(model, X[val_idx], device)))
        val_ba = balanced_accuracy_score(y[val_idx], prob > 0.5)
        val_loss = log_loss(y[val_idx], prob, labels=[0, 1])
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "val_loss": val_loss, "val_balanced_accuracy": val_ba, "kept": False})
        # The kept head is the one with the lowest validation loss
        if val_loss < best_loss:
            best_loss, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
            history[-1]["kept"] = True
        print(f"  epoch {epoch:3d}  train_loss={history[-1]['train_loss']:.4f}  "
              f"val_loss={val_loss:.4f}  val_bal_acc={val_ba:.4f}")
        # Early stopping follows the validation balanced accuracy
        if val_ba > best_ba + C.MLP_MIN_DELTA:
            best_ba, waited = val_ba, 0
        else:
            waited += 1
        if waited >= C.MLP_PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# Each test donor as a whole: the mean PD probability of its cells calls it PD
# or control. With two donors of each diagnosis in test, this table says
# whether the result holds for every donor or comes from one of them
def donor_table(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions[C.SPLIT_COL] == "test"]
    # observed=True: donor is a categorical column that also lists the donors of train and validation
    donors = test.groupby(C.DONOR_COL, observed=True).agg(
        diagnosis=(C.LABEL_COL, "first"), n_cells=("prob_pd", "size"),
        mean_prob_pd=("prob_pd", "mean"), frac_called_pd=("prob_pd", lambda p: (p > 0.5).mean()))
    donors["called"] = np.where(donors["mean_prob_pd"] > 0.5, C.PD_LABEL, "control")
    donors["correct"] = donors["called"] == donors["diagnosis"]
    return donors.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", choices=sorted(C.DATASET_DIRS), required=True)
    parser.add_argument("--model", choices=C.MODELS, required=True)
    parser.add_argument("--seed", type=int, default=C.SEED)
    args = parser.parse_args()

    emb = ad.read_h5ad(C.embeddings_path(args.dataset, args.model))
    X = np.asarray(emb.X, dtype=np.float32)
    y = (emb.obs[C.LABEL_COL] == C.PD_LABEL).to_numpy().astype(np.int64)
    split = emb.obs[C.SPLIT_COL].to_numpy()
    rng = np.random.default_rng(args.seed)
    train_idx = balance(y, np.flatnonzero(split == "train"), rng)
    val_idx, test_idx = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    print(f"Dataset {args.dataset}, {args.model}: train {len(train_idx)} (balanced), "
          f"val {len(val_idx)}, test {len(test_idx)} cells")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = train(X, y, train_idx, val_idx, device, args.seed)

    logits = predict_logits(model, X, device)
    predictions = emb.obs[[C.DONOR_COL, C.LABEL_COL, C.SPLIT_COL]].assign(
        logit=logits, prob_pd=1 / (1 + np.exp(-logits)))
    metrics = {"dataset": args.dataset, "model": args.model, "seed": args.seed,
               "best_epoch": int(history.loc[history["kept"], "epoch"].iloc[-1]),
               "n_train_cells_balanced": int(len(train_idx)),
               "test": cell_metrics(y[test_idx], predictions["prob_pd"].to_numpy()[test_idx]),
               "val": cell_metrics(y[val_idx], predictions["prob_pd"].to_numpy()[val_idx])}
    donors = donor_table(predictions)
    metrics["test"]["donors_correct"] = f"{int(donors['correct'].sum())}/{len(donors)}"

    out_dir = C.mlp_dir(args.dataset, args.model) / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same format as pipeline_unified's best_mlp.pt: the weights and the shape to rebuild the head
    torch.save({"state_dict": model.state_dict(),
                "config": {"input_dim": X.shape[1], "n_classes": 1,
                           **{k: C.MLP_PARAMS[k] for k in ("n_layers", "hidden_size", "dropout",
                                                           "activation", "mlp_type")}}},
               out_dir / "best_mlp.pt")
    history.to_csv(out_dir / "training_history.csv", index=False)
    predictions.rename_axis("barcode").to_csv(out_dir / "predictions.csv")
    donors.to_csv(out_dir / "test_donors.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    test = metrics["test"]
    lo, hi = test["balanced_accuracy_ci95"]
    print(f"TEST balanced accuracy {test['balanced_accuracy']:.3f} [{lo:.3f}, {hi:.3f}] 95% CI, "
          f"ROC AUC {test['roc_auc']:.3f}, donors correct {test['donors_correct']}")
    print(donors.to_string(index=False))
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
