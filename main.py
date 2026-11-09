import random
import numpy as np
import torch
from torch import nn

from vis import plot_loss_curve, plot_pred, eval
from dataset import load_df, window_dataset, make_frame, split_dataset
from model import RASATTCN
from baseline.rnn import GRU, LSTM
from baseline.tcn import TCN
from baseline.delaynet import DelayNet
from baseline.apcr_tcn import APCR_TCN

# 日级单步预测：过去 7 个自然日预测下一日
FREQ = "d"
SEQ_LEN = 7
P_COLS = ["lag1_z", "lag7_z"]
TRAIN_FRAC = 0.70
N_SEEDS = 5
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DATASET_NAME = "uci"

# 本轮实验开关。先完成轻量对照，避免一次性引入过多实验组合。
RUN_REGULARIZATION = True
RUN_FEATURE_ABLATION = True
RUN_SIMPLE_BASELINES = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_model(model, x, p, meta):
    if isinstance(model, RASATTCN):
        return model(x, p, meta)
    if isinstance(model, TCN) and model.in_ch > 1:
        return model(x)
    return model(x[:, :, 0])


def train(train_ds, val_ds, model, weight_decay=0.0, feature_indices=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val, best_epoch, best_train, wait, best_state = float("inf"), -1, float("inf"), 0, None

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y, p, meta in train_loader:
            x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
            if feature_indices is not None:
                x = x[:, :, feature_indices]
            optimizer.zero_grad()
            out = forward_model(model, x, p, meta)
            pred = out["pred"] if isinstance(out, dict) else out
            loss = criterion(pred, y.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))
        train_losses.append(train_loss)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y, p, meta in val_loader:
                x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
                if feature_indices is not None:
                    x = x[:, :, feature_indices]
                out = forward_model(model, x, p, meta)
                pred = out["pred"] if isinstance(out, dict) else out
                val_total += criterion(pred, y.squeeze(-1)).item()
        val_loss = val_total / max(1, len(val_loader))
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, best_train, wait = val_loss, epoch + 1, train_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d}: train={train_loss:.5f}, val={val_loss:.5f}")
        if wait >= PATIENCE:
            print(f"Early stop; best epoch={best_epoch}, val_loss={best_val:.5f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch, best_train, best_val


def test(test_ds, model, feature_indices=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.to(device).eval()
    gts, preds = [], []
    diagnostic = {k: [] for k in ["y_base", "R_t", "G_t", "delta"]}
    with torch.no_grad():
        for x, y, p, meta in loader:
            x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
            if feature_indices is not None:
                x = x[:, :, feature_indices]
            out = forward_model(model, x, p, meta)
            pred = out["pred"] if isinstance(out, dict) else out
            gts.append(y.cpu().numpy().reshape(-1))
            preds.append(pred.cpu().numpy().reshape(-1))
            if isinstance(out, dict):
                for key in diagnostic:
                    diagnostic[key].append(out[key].cpu().numpy().reshape(-1))
    gt, pred = np.concatenate(gts), np.concatenate(preds)
    diagnostics = {k: np.concatenate(v) for k, v in diagnostic.items() if v}
    return gt, pred, diagnostics


def build_experiments(features):
    in_dim = len(features)
    all_idx = list(range(in_dim))
    idx = {name: i for i, name in enumerate(features)}
    specs = {
        "TCN": {"factory": lambda: TCN(in_ch=1)},
        "Multivariate_TCN": {"factory": lambda: TCN(in_ch=in_dim), "feature_indices": all_idx},
        "APCR_TCN": {"factory": lambda: APCR_TCN()},
        "DelayNet": {"factory": lambda: DelayNet()},
        "GRU": {"factory": lambda: GRU(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1)},
        "LSTM": {"factory": lambda: LSTM(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1)},
        "RASAT_TCN": {"factory": lambda: RASATTCN(in_dim=in_dim, variant="full"), "feature_indices": all_idx},
        "RASAT_no_reliability": {"factory": lambda: RASATTCN(in_dim=in_dim, variant="no_reliability"), "feature_indices": all_idx},
        "RASAT_no_gate": {"factory": lambda: RASATTCN(in_dim=in_dim, variant="no_gate"), "feature_indices": all_idx},
        "RASAT_no_periodic": {"factory": lambda: RASATTCN(in_dim=in_dim, variant="no_periodic"), "feature_indices": all_idx},
    }

    if RUN_REGULARIZATION:
        for label, dropout, wd in [
            ("dropout02_wd1e4", 0.2, 1e-4),
            ("dropout03_wd1e4", 0.3, 1e-4),
            ("dropout02_wd5e4", 0.2, 5e-4),
        ]:
            specs[f"RASAT_{label}"] = {
                "factory": lambda d=dropout: RASATTCN(in_dim=in_dim, variant="full", dropout=d),
                "feature_indices": all_idx, "weight_decay": wd,
            }

    if RUN_FEATURE_ABLATION:
        # 仅筛选 x 主输入通道；p/meta 辅助通道保持不变，名称明确标注 aux_fixed，避免误称为全模型单特征输入。
        groups = {
            "load_only": ["load"],
            "load_periodic": ["load", "dow_sin", "dow_cos", "month_sin", "month_cos"],
            "load_state": ["load", "delta", "delta2", "rolling_std_7"],
        }
        for label, names in groups.items():
            selected = [idx[n] for n in names if n in idx]
            specs[f"RASAT_x_{label}_aux_fixed"] = {
                "factory": lambda n=len(selected): RASATTCN(in_dim=n, variant="full"),
                "feature_indices": selected,
            }
    return specs


def simple_baseline_predictions(test_ds):
    """返回标准化空间预测：lag-1 persistence 与输入窗口 7 日均值。"""
    loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    persistence, moving_average, gts = [], [], []
    for x, y, p, meta in loader:
        gts.append(y.numpy().reshape(-1))
        # p 是目标时点的历史滞后特征；第 0 列 lag1_z 即前一日负荷。
        persistence.append(p[:, 0, 0].numpy().reshape(-1))
        # load 特征使用与 target_z 相同的训练集标准化尺度。
        moving_average.append(x[:, :, 0].mean(dim=1).numpy().reshape(-1))
    return np.concatenate(gts), {
        "Persistence_lag1": np.concatenate(persistence),
        "MovingAverage_7d": np.concatenate(moving_average),
    }


def print_metric_summary(seed_metrics):
    print("\n===== Test metrics across seeds (original load unit; mean ± std) =====")
    for name, rows in seed_metrics.items():
        keys = ["MAE", "RMSE", "R2"]
        values = {k: np.asarray([r[k] for r in rows], dtype=float) for k in keys}
        print(
            f"{name}: " +
            " | ".join(f"{k}={values[k].mean():.5f} ± {values[k].std(ddof=1) if len(rows)>1 else 0.0:.5f}"
                       for k in keys)
        )


if __name__ == "__main__":
    print(f"Loading {DATASET_NAME}, frequency={FREQ}")
    _, dfd, dfh = load_df(ds_name=DATASET_NAME)
    source = dfh if FREQ == "h" else dfd
    frame, features, y_mean, y_std, thresholds = make_frame(source, freq=FREQ, fit_frac=TRAIN_FRAC)
    dataset = window_dataset(frame, features, seq_len=SEQ_LEN, p_cols=P_COLS)
    train_ds, val_ds, test_ds = split_dataset(dataset, ratios=(0.70, 0.15, 0.15))
    print(f"Samples total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Date span={source.index.min()} -> {source.index.max()}")
    print(f"Target mean/std (train only)={y_mean:.5f}/{y_std:.5f}")
    print(f"Thresholds={thresholds}")
    print(f"Feature order={features}")

    experiments = build_experiments(features)
    seed_results, seed_metrics, last_losses, mean_preds, common_gt = {}, {}, {}, {}, None
    rasat_diagnostics = {}
    for name, spec in experiments.items():
        seed_results[name], seed_metrics[name], seed_preds = [], [], []
        print(f"\n========== {name} ==========")
        for seed in range(N_SEEDS):
            set_seed(2026 + seed)
            model = spec["factory"]()
            model, tr_loss, va_loss, best_epoch, best_train, best_val = train(
                train_ds, val_ds, model,
                weight_decay=spec.get("weight_decay", 0.0),
                feature_indices=spec.get("feature_indices"),
            )
            gt, pred, diag = test(test_ds, model, feature_indices=spec.get("feature_indices"))
            common_gt = gt
            seed_preds.append(pred)
            metric = eval({name: gt}, {name: pred}, y_mean=y_mean, y_std=y_std)[name]
            seed_metrics[name].append(metric)
            if diag:
                diag["_final_pred"] = pred.copy()
                rasat_diagnostics[name] = diag
            seed_results[name].append({
                "last_train": tr_loss[-1], "last_val": va_loss[-1],
                "best_train": best_train, "best_val": best_val, "best_epoch": best_epoch,
            })
            if seed == N_SEEDS - 1:
                last_losses[name] = {"train": tr_loss, "val": va_loss}
        mean_preds[name] = np.mean(np.stack(seed_preds), axis=0)
        print(f"{name}: {N_SEEDS} seeds completed; best epochs={[x['best_epoch'] for x in seed_results[name]]}")

    if RUN_SIMPLE_BASELINES:
        gt, simple_preds = simple_baseline_predictions(test_ds)
        common_gt = gt
        for name, pred in simple_preds.items():
            metric = eval({name: gt}, {name: pred}, y_mean=y_mean, y_std=y_std)[name]
            seed_metrics[name] = [metric]
            mean_preds[name] = pred

    print("\n===== Seed summary (best and last train/val loss) =====")
    for name, rows in seed_results.items():
        print(name, rows)
    print_metric_summary(seed_metrics)

    print("\n===== RASAT internal diagnostics (standardized-space outputs; last seed) =====")
    for name, diag in rasat_diagnostics.items():
        print(name)
        for key, values in diag.items():
            if key == "_final_pred":
                continue
            print(f"  {key}: mean={values.mean():.4f}, std={values.std():.4f}, "
                  f"min={values.min():.4f}, max={values.max():.4f}")
        if "y_base" in diag:
            base_mae = np.mean(np.abs(common_gt - diag["y_base"]))
            final_mae = np.mean(np.abs(common_gt - diag["_final_pred"]))
            print(f"  y_base MAE(z)={base_mae:.4f}; final MAE(z)={final_mae:.4f}")

    plot_pred({"gt": common_gt}, mean_preds, y_mean=y_mean, y_std=y_std)
    plot_loss_curve({k: v["train"] for k, v in last_losses.items()}, title="Train Loss (last seed)")
    plot_loss_curve({k: v["val"] for k, v in last_losses.items()}, title="Validation Loss (last seed)")
