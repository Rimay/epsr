"""DSR 实验结果可视化。

函数：
    plot_training_history        各变体训练/验证损失曲线
    plot_prediction_comparison   多模型测试集预测曲线对比（含状态底色）
    plot_metrics_comparison      多模型多指标分组对比

命令行：python vis.py [results_dir]，默认 outputs/dsr 下最近修改的结果目录。
main.py 在训练测试结束后会自动调用以上函数。
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _save_or_show(fig, output_path):
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
    else:
        plt.show()
    plt.close(fig)


def load_variant_predictions(results_dir, seeds=None):
    """汇总各变体的测试集预测。

    每个变体取指定种子的 pred_{variant}_seed*.npz 平均（y_true/state 各种子相同）。
    返回 {variant: {"y_true", "y_pred", "state", "r2_mean"}}。
    """
    results_dir = Path(results_dir)
    cfg_path = results_dir / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    seeds = set(seeds or (cfg.get("seeds") if cfg else None) or [])

    by_variant = {}
    for f in results_dir.glob("pred_*.npz"):
        m = re.match(r"pred_(.+)_seed(\d+)\.npz", f.name)
        if not m:
            continue
        variant, seed = m.group(1), int(m.group(2))
        if seeds and seed not in seeds:
            continue  # 过滤掉配置外的一次性冒烟运行
        by_variant.setdefault(variant, []).append((seed, f))

    preds = {}
    for variant, files in sorted(by_variant.items()):
        stack, states = [], None
        for _, f in sorted(files):
            d = np.load(f)
            stack.append(d["y_pred"])
            states = d["state"]
        preds[variant] = {
            "y_true": np.load(files[0][1])["y_true"],
            "y_pred": np.mean(np.stack(stack), axis=0),  # 多种子平均预测
            "state": states,
            "r2_mean": float("nan"),
        }
    if cfg:
        r2 = pd.read_csv(results_dir / "metrics_by_seed.csv").groupby("model")["R2"].mean()
        for v in preds:
            preds[v]["r2_mean"] = float(r2.get(v, float("nan")))
    return preds


def _shade_states(ax, states, n):
    """为 Peak(1)/Ramp(2) 状态段绘制浅色背景。"""
    for sid, color in ((1, "orange"), (2, "steelblue")):
        idx = np.flatnonzero(states[:n] == sid)
        if len(idx) == 0:
            continue
        # 合并连续区段，避免逐点画大量矩形
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for run in np.split(idx, splits):
            ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color=color, alpha=0.10, lw=0)


def plot_training_history(csv_path, output_dir=None):
    """按变体绘制 train/val forecast MSE 曲线。"""
    df = pd.read_csv(csv_path)
    out = Path(output_dir or Path(csv_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    for name, g in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(g.epoch, g.train_forecast, label="Train")
        ax.plot(g.epoch, g.val_forecast, label="Validation")
        ax.set(title=f"{name}: forecast MSE", xlabel="Epoch", ylabel="MSE")
        ax.grid(True)
        ax.legend()
        _save_or_show(fig, out / f"loss_{name}.png")


def plot_prediction_comparison(results_dir, n_points=300, output_path=None):
    """多模型预测曲线对比：黑色粗线为真实值，彩色线为各变体多种子平均预测。

    背景色标注状态段：浅橙=Peak，浅蓝=Ramp；图例中附各变体平均 R²。
    """
    preds = load_variant_predictions(results_dir)
    if not preds:
        raise FileNotFoundError(f"No pred_*.npz found in {results_dir}")

    first = next(iter(preds.values()))
    n = min(n_points, len(first["y_true"]))
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_states(ax, first["state"], n)

    ax.plot(np.arange(n), first["y_true"][:n], color="black", lw=1.8,
            label="Actual", zorder=10)
    for (variant, data), color in zip(preds.items(), MODEL_COLORS):
        r2 = data["r2_mean"]
        r2_txt = f" (R²={r2:.3f})" if np.isfinite(r2) else ""
        ax.plot(np.arange(n), data["y_pred"][:n], color=color, lw=1.2,
                label=variant + r2_txt, alpha=0.9)

    ax.set(xlabel="Test sample", ylabel="Load",
           title=f"Test-set forecast comparison (mean over seeds, first {n} samples)")
    ax.grid(True, alpha=0.4)
    ax.legend(ncol=2, fontsize=9, loc="upper right")
    _save_or_show(fig, output_path or Path(results_dir) / "comparison_predictions.png")


def plot_metrics_comparison(results_dir, metrics=("MAE", "RMSE", "R2"), output_path=None):
    """多模型多指标分组柱状图（mean±std，跨种子）。"""
    df = pd.read_csv(Path(results_dir) / "metrics_by_seed.csv")
    models = sorted(df["model"].unique())
    agg = {m: df.groupby("model")[m].agg(["mean", "std"]).reindex(models) for m in metrics}

    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, metric in enumerate(metrics):
        s = agg[metric]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, s["mean"], width,
               yerr=s["std"].fillna(0), capsize=2, label=metric)

    ax.set(xticks=x, xticklabels=models, ylabel="value",
           title="Model comparison on test set (mean ± std over seeds)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.4)
    ax.legend()
    _save_or_show(fig, output_path or Path(results_dir) / "comparison_metrics.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", nargs="?", default=None,
                   help="结果目录；缺省取 outputs/dsr 下最近修改的子目录")
    p.add_argument("--n-points", type=int, default=300, help="预测对比图显示的样本数")
    p.add_argument("--no-show", action="store_true", help="只保存 PNG，不弹窗显示")
    args = p.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        base = Path("outputs/dsr")
        candidates = [d for d in base.iterdir() if d.is_dir()] if base.exists() else []
        results_dir = max(candidates, key=lambda d: d.stat().st_mtime)

    plot_training_history(results_dir / "training_history.csv")
    plot_prediction_comparison(results_dir, n_points=args.n_points)
    plot_metrics_comparison(results_dir)
    print(f"Saved figures to: {Path(results_dir).resolve()}")


if __name__ == "__main__":
    main()
