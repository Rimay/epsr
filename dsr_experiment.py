"""
DSR-Net experiments for the existing Rimay/epsr project.
Reuses dataset.py::load_df and make_frame; no changes to legacy RASAT/APCR code.
"""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

import dataset as data_module
from dataset import load_df, make_frame
from baseline.tcn import TCN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowStateDataset(Dataset):
    """One-step forecasting windows; state target is read only at t+1."""
    def __init__(self, frame, feature_cols, seq_len, freq):
        self.x, self.y, self.state = [], [], []
        x_all = frame[feature_cols].to_numpy(dtype=np.float32)
        y_all = frame["target_z"].to_numpy(dtype=np.float32)
        # make_frame computes these labels using training-segment quantiles only.
        if "is_peak" in frame and "is_ramp" in frame:
            peak = frame["is_peak"].to_numpy(dtype=np.int64)
            ramp = frame["is_ramp"].to_numpy(dtype=np.int64)
            state_all = np.zeros(len(frame), dtype=np.int64)  # 0 Normal/Valley
            state_all[ramp == 1] = 2
            state_all[peak == 1] = 1  # Peak takes precedence if both are true
        else:
            raw = frame["state"].astype(str).to_numpy()
            state_all = np.where(raw == "Peak", 1, np.where(raw == "Ramp", 2, 0))
        for end in range(seq_len, len(frame)):
            start = end - seq_len
            xw = x_all[start:end]
            target = y_all[end]
            if not (np.isfinite(xw).all() and np.isfinite(target)):
                continue
            self.x.append(xw)
            self.y.append(target)
            self.state.append(state_all[end])
        self.x = torch.tensor(np.asarray(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.y), dtype=torch.float32).view(-1, 1)
        self.state = torch.tensor(np.asarray(self.state), dtype=torch.long)
        self.freq = freq

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.state[index]


def chronological_split(ds, train_frac=0.70, val_frac=0.15):
    n = len(ds)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if min(n_train, n_val, n - n_train - n_val) <= 0:
        raise ValueError(f"Not enough samples for split: n={n}")
    return (Subset(ds, range(0, n_train)),
            Subset(ds, range(n_train, n_train + n_val)),
            Subset(ds, range(n_train + n_val, n)))


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, 3, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, 3, dilation)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.drop(self.act(self.conv1(x)))
        h = self.drop(self.act(self.conv2(h)))
        return self.act(h + self.skip(x))


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.3):
        super().__init__()
        blocks = []
        c = in_dim
        for dilation in (1, 2, 4):
            blocks.append(TCNBlock(c, hidden, dilation, dropout))
            c = hidden
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B,L,F] -> last causal representation [B,H]
        return self.net(x.transpose(1, 2))[:, :, -1]


class DSRNet(nn.Module):
    """
    Variants:
      state:      TCN + explicit state encoder + one regression head
      state_aux:  state + future-state auxiliary classifier + one regression head
      full:       state_aux + Normal/Peak/Ramp experts
      no_transition: full architecture, but CE loss disabled during training
      single_head: same auxiliary state learning as full, without experts
    """
    def __init__(self, in_dim, state_indices, hidden=32, state_dim=16,
                 dropout=0.3, variant="full"):
        super().__init__()
        self.variant = variant
        self.use_state = variant != "tcn"
        self.use_aux = variant in ("state_aux", "full", "no_transition", "single_head")
        self.use_experts = variant in ("full", "no_transition")
        self.temporal = TemporalEncoder(in_dim, hidden, dropout)
        state_in = len(state_indices)
        self.register_buffer("state_indices", torch.tensor(state_indices, dtype=torch.long))
        if self.use_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_in, 24), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(24, state_dim), nn.GELU()
            )
            fusion_dim = hidden + state_dim
        else:
            fusion_dim = hidden
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, hidden), nn.GELU(),
                                    nn.Dropout(dropout))
        self.state_classifier = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                              nn.Dropout(dropout), nn.Linear(32, 3))
        if self.use_experts:
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                              nn.Dropout(dropout), nn.Linear(32, 1))
                for _ in range(3)
            ])
        else:
            self.regressor = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        h_t = self.temporal(x)
        if self.use_state:
            # Explicit dynamic state: standardized level, first/second differences,
            # and rolling volatility at the final observed input time.
            s = x[:, -1, self.state_indices]
            h_s = self.state_encoder(s)
            h = self.fusion(torch.cat([h_t, h_s], dim=-1))
        else:
            h = self.fusion(h_t)
        logits = self.state_classifier(h)
        probs = torch.softmax(logits, dim=-1)
        if self.use_experts:
            expert_preds = torch.cat([expert(h) for expert in self.experts], dim=-1)
            pred = torch.sum(probs * expert_preds, dim=-1, keepdim=True)
        else:
            expert_preds = None
            pred = self.regressor(h)
        return {"pred": pred, "state_logits": logits, "state_probs": probs,
                "expert_preds": expert_preds}


def get_state_indices(feature_cols, freq):
    rolling = "rolling_std_24" if freq == "h" else "rolling_std_7"
    names = ["load", "delta", "delta2", rolling]
    missing = [n for n in names if n not in feature_cols]
    if missing:
        raise ValueError(f"Required dynamic-state features missing: {missing}; features={feature_cols}")
    return [feature_cols.index(n) for n in names]


def make_class_weights(train_subset, device):
    labels = torch.stack([train_subset[i][2] for i in range(len(train_subset))])
    counts = torch.bincount(labels, minlength=3).float()
    weights = torch.sqrt(counts.sum() / (3.0 * counts.clamp_min(1.0)))
    weights = weights / weights.mean()
    return weights.to(device), counts.tolist()


def compute_loss(output, y, state, model, variant, class_weights,
                 lambda_state=0.15, lambda_expert=0.20):
    forecast = nn.functional.mse_loss(output["pred"], y)
    total = forecast
    state_loss = torch.zeros((), device=y.device)
    expert_loss = torch.zeros((), device=y.device)
    if variant in ("state_aux", "full", "single_head"):
        state_loss = nn.functional.cross_entropy(
            output["state_logits"], state, weight=class_weights
        )
        total = total + lambda_state * state_loss
    elif variant == "no_transition":
        # Deliberately disable future-state supervision for this ablation.
        state_loss = nn.functional.cross_entropy(output["state_logits"], state)
    if output["expert_preds"] is not None:
        losses = []
        for k in range(3):
            mask = state == k
            if mask.any():
                losses.append(nn.functional.mse_loss(
                    output["expert_preds"][mask, k:k+1], y[mask]
                ))
        if losses:
            expert_loss = torch.stack(losses).mean()
            total = total + lambda_expert * expert_loss
    return total, forecast.detach(), state_loss.detach(), expert_loss.detach()


def train_model(model, train_ds, val_ds, args, device, variant, class_weights):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = np.zeros(4, dtype=float)
        n_batches = 0
        for x, y, state in train_loader:
            x, y, state = x.to(device), y.to(device), state.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, f_loss, s_loss, e_loss = compute_loss(
                out, y, state, model, variant, class_weights,
                args.lambda_state, args.lambda_expert
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sums += [loss.item(), f_loss.item(), s_loss.item(), e_loss.item()]
            n_batches += 1
        train_means = sums / max(n_batches, 1)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y, state in val_loader:
                x, y, state = x.to(device), y.to(device), state.to(device)
                out = model(x)
                val_sum += nn.functional.mse_loss(out["pred"], y).item()
                val_n += 1
        val_loss = val_sum / max(val_n, 1)
        history.append({
            "epoch": epoch, "train_total": train_means[0],
            "train_forecast": train_means[1], "train_state": train_means[2],
            "train_expert": train_means[3], "val_forecast": val_loss
        })
        if val_loss < best_val - args.min_delta:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"    epoch={epoch:03d} train={train_means[0]:.5f} "
                  f"val_mse={val_loss:.5f} best={best_val:.5f}@{best_epoch}")
        if wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, time.time() - start_time


def predict(model, ds, args, device):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ys, preds, states, probs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, state in loader:
            out = model(x.to(device))
            ys.append(y.numpy())
            preds.append(out["pred"].cpu().numpy())
            states.append(state.numpy())
            probs.append(out["state_probs"].cpu().numpy())
    return (np.concatenate(ys).reshape(-1), np.concatenate(preds).reshape(-1),
            np.concatenate(states).reshape(-1), np.concatenate(probs, axis=0))


def metric_row(y_true_z, y_pred_z, state, y_mean, y_std, model_name, seed,
               params, train_seconds, best_epoch, group="Overall"):
    y_true = y_true_z * y_std + y_mean
    y_pred = y_pred_z * y_std + y_mean
    if len(y_true) == 0:
        return None
    return {
        "model": model_name, "seed": seed, "group": group, "n": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "parameters": params, "train_seconds": train_seconds, "best_epoch": best_epoch
    }


def evaluate_run(model, test_ds, args, device, y_mean, y_std,
                 model_name, seed, train_seconds, best_epoch):
    yt, yp, states, state_probs = predict(model, test_ds, args, device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = [metric_row(yt, yp, states, y_mean, y_std, model_name, seed,
                       params, train_seconds, best_epoch, "Overall")]
    state_pred = state_probs.argmax(axis=1)
    rows[0]["state_accuracy"] = accuracy_score(states, state_pred)
    rows[0]["state_macro_f1"] = f1_score(states, state_pred, labels=[0, 1, 2],
                                           average="macro", zero_division=0)
    # Peak and Ramp use their target-state labels; Normal includes Valley.
    for label, code in (("Normal", 0), ("Peak", 1), ("Ramp", 2)):
        mask = states == code
        row = metric_row(yt[mask], yp[mask], states[mask], y_mean, y_std,
                         model_name, seed, params, train_seconds, best_epoch, label)
        if row is not None:
            rows.append(row)
    return rows, yt, yp, states


def parse_args():
    p = argparse.ArgumentParser(description="Run DSR-Net experiments in Rimay/epsr")
    p.add_argument("--dataset", choices=["uci", "GEFCom2014"], default="uci")
    p.add_argument("--freq", choices=["h", "d"], default="d")
    p.add_argument("--data-root", default=None,
                   help="Override dataset.DATA_PATH, e.g. C:/.../EPSR/data/")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lambda-state", type=float, default=0.15)
    p.add_argument("--lambda-expert", type=float, default=0.20)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    p.add_argument("--models", nargs="+",
                   default=["tcn", "state", "state_aux", "full", "no_transition", "single_head"],
                   choices=["tcn", "state", "state_aux", "full", "no_transition", "single_head"])
    p.add_argument("--output-dir", default="results/dsr_net")
    return p.parse_args()


def main():
    args = parse_args()
    if args.data_root:
        data_module.DATA_PATH = args.data_root.rstrip("/\\\\") + os.sep
    args.seq_len = args.seq_len or (168 if args.freq == "h" else 7)
    p_cols = ["lag24_z", "lag168_z"] if args.freq == "h" else ["lag1_z", "lag7_z"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; dataset={args.dataset}; freq={args.freq}; seq_len={args.seq_len}")
    df, dfd, dfh = load_df(ds_name=args.dataset)
    source = dfh if args.freq == "h" else dfd
    frame, feature_cols, y_mean, y_std, thresholds = make_frame(
        source, freq=args.freq, fit_frac=args.train_frac
    )
    ds = WindowStateDataset(frame, feature_cols, args.seq_len, args.freq)
    train_ds, val_ds, test_ds = chronological_split(ds, args.train_frac, args.val_frac)
    state_indices = get_state_indices(feature_cols, args.freq)
    print(f"Samples={len(ds)} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Features={feature_cols}; state_indices={state_indices}")
    print(f"Train-only thresholds={json.dumps(thresholds, ensure_ascii=False)}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows, histories = [], []
    for seed in args.seeds:
        print(f"\\n========== Seed {seed} ==========")
        set_seed(seed)
        class_weights, class_counts = make_class_weights(train_ds, device)
        print(f"Train state counts [Normal,Peak,Ramp]={class_counts}")
        for variant in args.models:
            set_seed(seed)  # paired initialization/order across model variants
            print(f"\\n--- {variant} ---")
            if variant == "tcn":
                model = TCN(in_ch=1).to(device)
                # Adapt the legacy univariate TCN to the same train/eval protocol.
                class LegacyWrapper(nn.Module):
                    def __init__(self, base):
                        super().__init__()
                        self.base = base
                    def forward(self, x):
                        pred = self.base(x[:, :, 0])
                        if pred.ndim == 1:
                            pred = pred.unsqueeze(-1)
                        z = torch.zeros((len(x), 3), device=x.device)
                        z[:, 0] = 1.0
                        return {"pred": pred, "state_logits": z,
                                "state_probs": torch.softmax(z, dim=-1),
                                "expert_preds": None}
                model = LegacyWrapper(model).to(device)
                variant_for_loss = "tcn"
            else:
                model = DSRNet(len(feature_cols), state_indices, args.hidden,
                               dropout=args.dropout, variant=variant).to(device)
                variant_for_loss = variant
            # The TCN baseline has no auxiliary objectives.
            if variant == "tcn":
                class_weights_run = class_weights
            else:
                class_weights_run = class_weights
            model, history, best_epoch, seconds = train_model(
                model, train_ds, val_ds, args, device, variant_for_loss, class_weights_run
            )
            histories.extend([{"seed": seed, "model": variant, **h} for h in history])
            rows, yt, yp, states = evaluate_run(
                model, test_ds, args, device, y_mean, y_std,
                variant, seed, seconds, best_epoch
            )
            all_rows.extend(rows)
            overall = next(r for r in rows if r["group"] == "Overall")
            print(f"TEST {variant}: MAE={overall['MAE']:.5f} RMSE={overall['RMSE']:.5f} "
                  f"R2={overall['R2']:.5f}; state_acc={overall['state_accuracy']:.4f}; "
                  f"state_macro_f1={overall['state_macro_f1']:.4f}; "
                  f"params={overall['parameters']}; time={seconds:.1f}s; best_epoch={best_epoch}")
            np.savez_compressed(
                os.path.join(args.output_dir, f"pred_{args.dataset}_{args.freq}_{variant}_seed{seed}.npz"),
                y_true_z=yt, y_pred_z=yp, state=states
            )
    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(args.output_dir, "metrics_by_seed.csv"), index=False)
    pd.DataFrame(histories).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)
    summary = (results.groupby(["model", "group"])
               .agg({c: ["mean", "std"] for c in ["MAE", "RMSE", "R2", "parameters", "train_seconds", "best_epoch"]})
               .reset_index())
    summary.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    print("\\n===== Mean ± std across seeds =====")
    for model_name in args.models:
        subset = results[(results.model == model_name) & (results.group == "Overall")]
        if len(subset):
            print(f"{model_name:16s} MAE={subset.MAE.mean():.5f}±{subset.MAE.std():.5f} "
                  f"RMSE={subset.RMSE.mean():.5f}±{subset.RMSE.std():.5f} "
                  f"R2={subset.R2.mean():.5f}±{subset.R2.std():.5f}")
    print(f"\\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
