import random
import numpy as np
import torch
from torch import nn

from vis import plot_loss_curve, plot_prediction, eval
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
N_SEEDS = 5                 # 正式实验建议 5 次；调试时可临时设为 1
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DATASET_NAME = "uci"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_model(model, x, p, meta):
    if isinstance(model, RASATTCN):
        return model(x, p, meta)
    if isinstance(model, TCN) and model.net[0].conv1.conv.in_channels > 1:
        return model(x)
    # 其余原始基线保持单变量输入，避免悄悄改变既有基线定义
    return model(x[:, :, 0])


def train(train_ds, val_ds, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val, best_epoch, wait, best_state = float("inf"), -1, 0, None

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y, p, meta in train_loader:
            x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
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
                out = forward_model(model, x, p, meta)
                pred = out["pred"] if isinstance(out, dict) else out
                val_total += criterion(pred, y.squeeze(-1)).item()
        val_loss = val_total / max(1, len(val_loader))
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, wait = val_loss, epoch + 1, 0
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
    return model, train_losses, val_losses, best_epoch


def test(test_ds, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.to(device).eval()
    truths, predictions = [], []
    diagnostic = {k: [] for k in ["y_base", "R_t", "G_t", "delta"]}
    with torch.no_grad():
        for x, y, p, meta in loader:
            x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
            out = forward_model(model, x, p, meta)
            pred = out["pred"] if isinstance(out, dict) else out
            truths.append(y.cpu().numpy().reshape(-1))
            predictions.append(pred.cpu().numpy().reshape(-1))
            if isinstance(out, dict):
                for key in diagnostic:
                    diagnostic[key].append(out[key].cpu().numpy().reshape(-1))
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    diagnostics = {k: np.concatenate(v) for k, v in diagnostic.items() if v}
    return truth, prediction, diagnostics


def build_models(in_dim):
    return {
        "TCN": lambda: TCN(in_ch=1),
        "Multivariate_TCN": lambda: TCN(in_ch=in_dim),
        "APCR_TCN": lambda: APCR_TCN(),
        "DelayNet": lambda: DelayNet(),
        "GRU": lambda: GRU(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1),
        "LSTM": lambda: LSTM(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1),
        "RASAT_TCN": lambda: RASATTCN(in_dim=in_dim, variant="full"),
        "RASAT_no_reliability": lambda: RASATTCN(in_dim=in_dim, variant="no_reliability"),
        "RASAT_no_gate": lambda: RASATTCN(in_dim=in_dim, variant="no_gate"),
        "RASAT_no_periodic": lambda: RASATTCN(in_dim=in_dim, variant="no_periodic"),
    }


if __name__ == "__main__":
    print(f"Loading {DATASET_NAME}, frequency={FREQ}")
    _, daily, hourly = load_df(ds_name=DATASET_NAME)
    source = hourly if FREQ == "h" else daily
    frame, features, y_mean, y_std, thresholds = make_frame(
        source, freq=FREQ, fit_frac=TRAIN_FRAC
    )
    dataset = window_dataset(frame, features, seq_len=SEQ_LEN, p_cols=P_COLS)
    train_ds, val_ds, test_ds = split_dataset(dataset, ratios=(0.70, 0.15, 0.15))
    print(f"Samples total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Date span={source.index.min()} -> {source.index.max()}")
    print(f"Target mean/std (train only)={y_mean:.5f}/{y_std:.5f}")
    print(f"Thresholds={thresholds}")

    factories = build_models(len(features))
    seed_results, last_losses, mean_predictions, common_truth = {}, {}, {}, None
    rasat_diagnostics = {}
    for name, factory in factories.items():
        seed_results[name] = []
        seed_predictions = []
        print(f"\n========== {name} ==========")
        for seed in range(N_SEEDS):
            set_seed(2026 + seed)
            model = factory()
            model, tr_loss, va_loss, best_epoch = train(train_ds, val_ds, model)
            truth, prediction, diag = test(test_ds, model)
            common_truth = truth
            seed_predictions.append(prediction)
            # Each run is evaluated in original physical units.
            eval({name: truth}, {name: prediction}, y_mean=y_mean, y_std=y_std)
            if diag:
                rasat_diagnostics[name] = diag
            seed_results[name].append((tr_loss[-1], va_loss[-1], best_epoch))
            if seed == N_SEEDS - 1:
                last_losses[name] = {"train": tr_loss, "val": va_loss}
        mean_predictions[name] = np.mean(np.stack(seed_predictions), axis=0)
        print(f"{name}: {N_SEEDS} seeds completed; best epochs={[x[2] for x in seed_results[name]]}")

    print("\n===== Seed summary (best epoch; last train/val loss) =====")
    for name, rows in seed_results.items():
        print(name, rows)

    print("\n===== RASAT internal diagnostics (standardized-space outputs) =====")
    for name, diag in rasat_diagnostics.items():
        print(name)
        for key, values in diag.items():
            print(f"  {key}: mean={values.mean():.4f}, std={values.std():.4f}, "
                  f"min={values.min():.4f}, max={values.max():.4f}")
        if "y_base" in diag:
            base_mae = np.mean(np.abs(common_truth - diag["y_base"]))
            final_mae = np.mean(np.abs(common_truth - mean_predictions[name]))
            print(f"  y_base MAE(z)={base_mae:.4f}; final/seed-mean MAE(z)={final_mae:.4f}")

    # 图中展示各随机种子预测均值，纵轴恢复到原始负荷单位
    plot_prediction({"truth": common_truth}, mean_predictions, y_mean=y_mean, y_std=y_std)
    # 训练曲线使用最后一个 seed，分别绘制 train/validation
    plot_loss_curve({k: v["train"] for k, v in last_losses.items()}, title="Train Loss (last seed)")
    plot_loss_curve({k: v["val"] for k, v in last_losses.items()}, title="Validation Loss (last seed)")
