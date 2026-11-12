"""
流程：load_df 读原始数据 → make_frame 构造防泄露特征 → WindowDataset 滑窗
→ chronological_split 按时间切分 → 逐变体/逐种子训练（验证集早停）→
测试集预测并反标准化 → 整体与分状态指标 → 结果与对比图落盘 output-dir。
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import load_df, make_frame, WindowDataset, chronological_split
from model import DSRNet
from vis import plot_metrics_comparison, plot_prediction_comparison, plot_training_history

def seed_all(seed):
    """固定 random/numpy/torch 种子，保证同一种子结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["uci", "GEFCom2014"], default="uci", help="数据集名称")
    p.add_argument("--freq", choices=["h", "d"], default="d", help="采样频率：h=小时级，d=天级")
    p.add_argument("--data-root", default="data", help="原始数据根目录")
    p.add_argument("--output-dir", default="outputs/dsr", help="结果输出目录")
    p.add_argument("--seq-len", type=int, default=None, help="历史窗口长度；缺省 h=168、d=28")
    p.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    p.add_argument("--patience", type=int, default=15, help="早停耐心轮数")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lambda-state", type=float, default=0.15, help="状态分类辅助损失权重")
    p.add_argument("--lambda-expert", type=float, default=0.2, help="专家预测辅助损失权重")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55],
                   help="随机种子列表，每个种子各训练一遍")
    p.add_argument("--models", nargs="+", choices=sorted(DSRNet.VARIANTS),
                   default=["tcn", "state", "state_aux", "full", "no_transition"],
                   help="要训练的模型变体")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def select_inputs(x, variant):
    """tcn 变体只用 load 单通道，其余变体使用全部特征通道。"""
    return x[:, :, :1] if variant == "tcn" else x


def run_epoch(model, data, variant, optimizer, device, args, training):
    """跑一个 epoch，返回各项平均损失。

    损失构成：
        forecast：主任务 MSE，始终计入；
        state   ：状态分类交叉熵，state_aux/full 变体计入总损失（权重 lambda_state）；
        expert  ：各状态专家在自己状态样本上的 MSE，full/no_transition 计入（权重 lambda_expert）。
    """
    model.train(training)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    sums = {"total": 0.0, "forecast": 0.0, "state": 0.0, "expert": 0.0}
    n_batches = 0

    for x, y, state in data:
        x = select_inputs(x.to(device), variant)
        y, state = y.to(device), state.to(device)
        with torch.set_grad_enabled(training):
            out = model(x)
            forecast_loss = mse(out["pred"], y)
            state_loss = (ce(out["state_logits"], state)
                          if "state_logits" in out else forecast_loss.new_zeros(()))
            expert_loss = forecast_loss.new_zeros(())
            if "expert_preds" in out:
                # 每个专家只在自己对应状态的样本上计算损失
                terms = [mse(out["expert_preds"][state == k, k:k + 1], y[state == k])
                         for k in range(3) if (state == k).any()]
                if terms:
                    expert_loss = torch.stack(terms).mean()

            loss = forecast_loss
            if variant in {"state_aux", "full"}:
                loss = loss + args.lambda_state * state_loss
            if variant in {"full", "no_transition"}:
                loss = loss + args.lambda_expert * expert_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for key, value in (("total", loss), ("forecast", forecast_loss),
                           ("state", state_loss), ("expert", expert_loss)):
            sums[key] += float(value.detach())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in sums.items()}


@torch.no_grad()
def predict(model, data, variant, device, meta):
    """在给定数据集上预测，并把标准化输出反变换回原始量纲。"""
    model.eval()
    ys, ps, ss, probs = [], [], [], []
    for x, y, state in data:
        out = model(select_inputs(x.to(device), variant))
        ys.append(y.numpy().ravel())
        ps.append(out["pred"].cpu().numpy().ravel())
        ss.append(state.numpy().ravel())
        if "state_probs" in out:
            probs.append(out["state_probs"].cpu().numpy())

    y_true = np.concatenate(ys) * meta["y_std"] + meta["y_mean"]
    y_pred = np.concatenate(ps) * meta["y_std"] + meta["y_mean"]
    return {"y_true": y_true, "y_pred": y_pred, "state": np.concatenate(ss),
            "state_probs": np.concatenate(probs) if probs else None}


def regression_metrics(y_true, y_pred):
    """常规回归指标：MAE / RMSE / R² / MAPE(%)。"""
    err = y_pred - y_true
    denom = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
        "MAPE_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-6)) * 100),
    }


def score_result(result, variant, seed, best_epoch, best_val):
    """汇总整体指标、分状态（Normal/Peak/Ramp）指标与状态分类准确率。"""
    metrics = regression_metrics(result["y_true"], result["y_pred"])
    metrics.update(model=variant, seed=seed, best_epoch=best_epoch, best_val_mse=best_val)

    for label, state_id in (("Normal", 0), ("Peak", 1), ("Ramp", 2)):
        mask = result["state"] == state_id
        if mask.any():
            sub = regression_metrics(result["y_true"][mask], result["y_pred"][mask])
            metrics[label + "_MAE"] = sub["MAE"]
            metrics[label + "_RMSE"] = sub["RMSE"]
        else:
            metrics[label + "_MAE"] = metrics[label + "_RMSE"] = float("nan")

    if result["state_probs"] is not None:
        pred_state = result["state_probs"].argmax(1)
        true_state = result["state"]
        metrics["state_accuracy"] = float(np.mean(pred_state == true_state))
        recalls = [np.mean(pred_state[true_state == k] == k)
                   for k in range(3) if np.any(true_state == k)]
        metrics["state_macro_recall"] = float(np.mean(recalls))
    return metrics


def train_variant(variant, seed, parts, features, meta, args, out_dir, device):
    """训练单个变体（指定种子）：验证集早停选最优权重，测试集评估并落盘。"""
    seed_all(seed)
    state_cols = ["load", "delta", "delta2",
                  "rolling_std_24" if args.freq == "h" else "rolling_std_7"]
    state_indices = [features.index(c) for c in state_cols] if variant != "tcn" else [0]
    in_dim = 1 if variant == "tcn" else len(features)
    model = DSRNet(in_dim, state_indices, variant=variant, dropout=args.dropout).to(device)

    loaders = []
    for i, ds in enumerate(parts):
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0),  # 仅训练集打乱
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders[0], variant, optimizer, device, args, training=True)
        val_loss = run_epoch(model, loaders[1], variant, optimizer, device, args, training=False)
        history.append({"model": variant, "seed": seed, "epoch": epoch,
                        **{f"train_{k}": v for k, v in train_loss.items()},
                        **{f"val_{k}": v for k, v in val_loss.items()}})

        # 以验证集 forecast MSE 为早停依据，保留最优权重
        if val_loss["forecast"] < best_val - 1e-7:
            best_val, best_epoch, wait = val_loss["forecast"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"{variant} seed={seed} epoch={epoch:03d} "
                  f"train={train_loss['forecast']:.5f} val={val_loss['forecast']:.5f}")
        if wait >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint for {variant}, seed {seed}")
    model.load_state_dict(best_state)

    result = predict(model, loaders[2], variant, device, meta)
    metrics = score_result(result, variant, seed, best_epoch, best_val)
    np.savez_compressed(out_dir / f"pred_{variant}_seed{seed}.npz",
                        **{k: v for k, v in result.items() if v is not None})
    torch.save({"state_dict": model.state_dict(), "variant": variant, "seed": seed,
                "features": features, "metadata": meta},
               out_dir / f"best_{variant}_seed{seed}.pt")
    return metrics, history


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs/patience/batch-size must be positive")

    seq_len = args.seq_len or (168 if args.freq == "h" else 28)
    if seq_len < 2:
        raise ValueError("seq-len must be >= 2")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.freq}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, dfd, dfh = load_df(args.dataset, args.data_root)
    frame, features, meta = make_frame(dfh if args.freq == "h" else dfd, args.freq)
    dataset = WindowDataset(frame, features, seq_len)
    parts = chronological_split(dataset, meta["train_end"], meta["val_end"])
    print(f"device={device}; windows={len(dataset)}; split=" + "/".join(str(len(x)) for x in parts))
    print(f"features={features}; peak={meta['peak_threshold']:.5g}; ramp={meta['ramp_threshold']:.5g}")

    all_metrics, history = [], []
    for variant in args.models:
        for seed in args.seeds:
            metrics, hist = train_variant(variant, seed, parts, features, meta, args, out_dir, device)
            all_metrics.append(metrics)
            history.extend(hist)
            print(f"TEST {variant} seed={seed}: MAE={metrics['MAE']:.5f} "
                  f"RMSE={metrics['RMSE']:.5f} R2={metrics['R2']:.5f}")

    df = pd.DataFrame(all_metrics)
    df.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.groupby("model")[list(numeric_cols)].agg(["mean", "std"]).to_csv(out_dir / "metrics_summary.csv")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    config = vars(args).copy()
    config.update(seq_len=seq_len, device=str(device), features=features, metadata=meta)
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 调用 vis.py 完成可视化（需先写好 experiment_config.json，绘图时据此过滤种子）----
    plot_training_history(out_dir / "training_history.csv")
    plot_prediction_comparison(out_dir)
    plot_metrics_comparison(out_dir)
    print(f"Results: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
