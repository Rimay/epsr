"""Reproducible DSR-Net and baseline experiments for EPSR.

Examples:
  python main.py --dataset uci --freq d --models tcn state state_aux full --seeds 11 22 33
  python main.py --dataset GEFCom2014 --freq h --data-root ./data --epochs 3 --seeds 11
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import load_df, make_frame, WindowDataset, split_dataset
from model import DSRNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["uci", "GEFCom2014"], default="uci")
    p.add_argument("--freq", choices=["h", "d"], default="d")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default="outputs/dsr")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lambda-state", type=float, default=0.15)
    p.add_argument("--lambda-expert", type=float, default=0.20)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    p.add_argument("--models", nargs="+",
                   choices=["tcn", "state", "state_aux", "full"],
                   default=["tcn", "state", "state_aux", "full"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def get_state_indices(features, freq):
    wanted = ["load", "delta", "delta2", "rolling_std_24" if freq == "h" else "rolling_std_7"]
    missing = [name for name in wanted if name not in features]
    if missing:
        raise ValueError(f"State descriptor columns not found: {missing}")
    return [features.index(name) for name in wanted]


def make_model(name, in_dim, state_indices, dropout):
    return DSRNet(in_dim=in_dim, state_indices=state_indices,
                  dropout=dropout, variant=name)


def subset_targets(subset):
    return subset.dataset.states[subset.indices]


def calculate_metrics(y_true_z, y_pred_z, y_mean, y_std):
    y_true = y_true_z * y_std + y_mean
    y_pred = y_pred_z * y_std + y_mean
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def state_metrics(y_true, y_pred, n_states=3):
    cm = np.zeros((n_states, n_states), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_states and 0 <= b < n_states:
            cm[int(a), int(b)] += 1
    recalls = []
    for i in range(n_states):
        recalls.append(cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan)
    return {"state_accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "state_macro_recall": float(np.nanmean(recalls)), "state_confusion": cm.tolist()}


def run_epoch(model, loader, optimizer, device, args, training):
    model.train(training)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    totals = {"loss": 0.0, "forecast": 0.0, "state": 0.0, "expert": 0.0}
    n_batches = 0
    for x, y, state in loader:
        x, y, state = x.to(device), y.to(device), state.to(device)
        with torch.set_grad_enabled(training):
            out = model(x)
            forecast_loss = mse(out["pred"], y)
            state_loss = (ce(out["state_logits"], state)
                          if "state_logits" in out else forecast_loss.new_zeros(()))
            expert_loss = forecast_loss.new_zeros(())
            if "expert_preds" in out:
                # Train each expert only on examples belonging to its target regime.
                terms = []
                for k in range(3):
                    mask = state == k
                    if mask.any():
                        terms.append(mse(out["expert_preds"][mask, k:k+1], y[mask]))
                if terms:
                    expert_loss = torch.stack(terms).mean()
            loss = forecast_loss
            if "state_logits" in out:
                loss = loss + args.lambda_state * state_loss
            if "expert_preds" in out:
                loss = loss + args.lambda_expert * expert_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["forecast"] += float(forecast_loss.detach())
        totals["state"] += float(state_loss.detach())
        totals["expert"] += float(expert_loss.detach())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds, states, state_preds, probs = [], [], [], [], []
    for x, y, state in loader:
        out = model(x.to(device))
        ys.append(y.numpy().reshape(-1))
        preds.append(out["pred"].cpu().numpy().reshape(-1))
        states.append(state.numpy().reshape(-1))
        if "state_probs" in out:
            probs.append(out["state_probs"].cpu().numpy())
            state_preds.append(out["state_probs"].argmax(dim=1).cpu().numpy())
    result = {"y_true_z": np.concatenate(ys), "y_pred_z": np.concatenate(preds),
              "state": np.concatenate(states)}
    if probs:
        result["state_probs"] = np.concatenate(probs)
        result["state_pred"] = np.concatenate(state_preds)
    return result


def train_one(model_name, train_ds, val_ds, test_ds, in_dim, state_indices,
              y_mean, y_std, seed, args, output_dir, device):
    set_seed(seed)
    model = make_model(model_name, in_dim, state_indices, args.dropout).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, args, True)
        va = run_epoch(model, val_loader, optimizer, device, args, False)
        history.append({"seed": seed, "model": model_name, "epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        if va["forecast"] < best_loss - 1e-6:
            best_loss, best_epoch, wait = va["forecast"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{model_name} seed={seed} epoch={epoch:03d} "
                  f"train={tr['forecast']:.5f} val={va['forecast']:.5f}")
        if wait >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    result = predict(model, test_loader, device)
    metrics = calculate_metrics(result["y_true_z"], result["y_pred_z"], y_mean, y_std)
    metrics.update({"model": model_name, "seed": seed, "best_epoch": best_epoch,
                    "best_val_mse": best_loss})
    if "state_pred" in result:
        metrics.update(state_metrics(result["state"], result["state_pred"]))
    np.savez_compressed(output_dir / f"pred_{model_name}_seed{seed}.npz", **result)
    return metrics, history


def main():
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch-size must be positive")
    seq_len = args.seq_len or (168 if args.freq == "h" else 7)
    output_dir = Path(args.output_dir) / f"{args.dataset}_{args.freq}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}; dataset={args.dataset}; freq={args.freq}; seq_len={seq_len}")

    _, daily, hourly = load_df(args.dataset, args.data_root)
    source = hourly if args.freq == "h" else daily
    frame, features, y_mean, y_std, thresholds = make_frame(source, args.freq, fit_frac=0.70)
    ds = WindowDataset(frame, features, seq_len=seq_len)
    if len(ds) < 30:
        raise ValueError(f"Too few valid windows ({len(ds)}). Check data, frequency and seq_len.")
    train_ds, val_ds, test_ds = split_dataset(ds, (0.70, 0.15, 0.15))
    state_indices = get_state_indices(features, args.freq)
    print(f"samples={len(ds)} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"features={features}; state_indices={state_indices}; thresholds={thresholds}")

    all_metrics, all_history = [], []
    for model_name in args.models:
        for seed in args.seeds:
            metrics, history = train_one(
                model_name, train_ds, val_ds, test_ds, len(features), state_indices,
                y_mean, y_std, seed, args, output_dir, device
            )
            all_metrics.append(metrics)
            all_history.extend(history)
            print(f"TEST {model_name} seed={seed}: MAE={metrics['MAE']:.5f} "
                  f"RMSE={metrics['RMSE']:.5f} R2={metrics['R2']:.5f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary = metrics_df.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    summary.to_csv(output_dir / "metrics_summary.csv")
    pd.DataFrame(all_history).to_csv(output_dir / "training_history.csv", index=False)
    metadata = {"dataset": args.dataset, "freq": args.freq, "seq_len": seq_len,
                "features": features, "state_features": ["load", "delta", "delta2",
                "rolling_std_24" if args.freq == "h" else "rolling_std_7"],
                "thresholds": thresholds, "y_mean": y_mean, "y_std": y_std,
                "seeds": args.seeds, "models": args.models, "device": str(device)}
    (output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
