"""DSR experiment visualizations with per-model panels and seed uncertainty bands."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_variant_predictions(results_dir, seeds=None):
    """Load same-split test predictions and aggregate across random seeds."""
    results_dir = Path(results_dir)
    cfg_path = results_dir / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    selected = set(seeds or cfg.get("seeds", []))
    by_variant = {}

    for f in results_dir.glob("pred_*.npz"):
        match = re.match(r"pred_(.+)_seed(\d+)\.npz", f.name)
        if not match:
            continue
        variant, seed = match.group(1), int(match.group(2))
        if selected and seed not in selected:
            continue
        by_variant.setdefault(variant, []).append((seed, f))

    preds = {}
    for variant, files in sorted(by_variant.items()):
        files = sorted(files)
        arrays = [np.load(path) for _, path in files]
        y_true = arrays[0]["y_true"]
        states = arrays[0]["state"]
        for arr in arrays[1:]:
            if not np.allclose(arr["y_true"], y_true):
                raise ValueError(f"Different test targets found for {variant}; cannot average seeds")
            if not np.array_equal(arr["state"], states):
                raise ValueError(f"Different state labels found for {variant}; cannot average seeds")
        stack = np.stack([arr["y_pred"] for arr in arrays])
        preds[variant] = {
            "y_true": y_true,
            "y_pred": stack.mean(axis=0),
            "seed_std": stack.std(axis=0),
            "state": states,
            "n_seeds": len(arrays),
            "r2_mean": float("nan"),
        }

    metric_path = results_dir / "metrics_by_seed.csv"
    if metric_path.exists():
        frame = pd.read_csv(metric_path)
        if {"model", "R2"}.issubset(frame.columns):
            r2 = frame.groupby("model")["R2"].mean()
            for variant in preds:
                preds[variant]["r2_mean"] = float(r2.get(variant, float("nan")))
    return preds


def _shade_states(ax, states, n):
    for sid, color, label in ((1, "orange", "Peak"), (2, "steelblue", "Ramp")):
        idx = np.flatnonzero(states[:n] == sid)
        if not len(idx):
            continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for j, run in enumerate(np.split(idx, splits)):
            ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color=color,
                       alpha=0.10, lw=0, label=label if j == 0 else None)


def plot_training_history(csv_path, output_dir=None):
    df = pd.read_csv(csv_path)
    out = Path(output_dir or Path(csv_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    for name, group in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for seed, part in group.groupby("seed"):
            ax.plot(part.epoch, part.train_forecast, alpha=0.35, lw=1)
            ax.plot(part.epoch, part.val_forecast, alpha=0.65, lw=1,
                    linestyle="--", label=f"seed {seed} val")
        ax.set(title=f"{name}: train/validation forecast MSE",
               xlabel="Epoch", ylabel="MSE")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        _save(fig, out / f"loss_{name}.png")


def plot_prediction_comparison(results_dir, n_points=300, output_path=None):
    """One panel per model; shade the across-seed prediction standard deviation."""
    results_dir = Path(results_dir)
    preds = load_variant_predictions(results_dir)
    if not preds:
        raise FileNotFoundError(f"No pred_*.npz found in {results_dir}")

    variants = list(preds)
    n = min(n_points, min(len(v["y_true"]) for v in preds.values()))
    rows, cols = int(np.ceil(len(variants) / 2)), 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.0 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for i, variant in enumerate(variants):
        ax = axes.flat[i]
        item = preds[variant]
        x = np.arange(n)
        _shade_states(ax, item["state"], n)
        ax.plot(x, item["y_true"][:n], color="black", lw=1.6,
                label="Actual", zorder=5)
        ax.plot(x, item["y_pred"][:n], color=MODEL_COLORS[i % len(MODEL_COLORS)],
                lw=1.25, label="Mean prediction")
        band = item["seed_std"][:n]
        ax.fill_between(x, item["y_pred"][:n] - band,
                        item["y_pred"][:n] + band,
                        color=MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.16,
                        label="±1 seed std")
        r2 = item["r2_mean"]
        r2_text = f" | mean R²={r2:.3f}" if np.isfinite(r2) else ""
        ax.set_title(f"{variant} (n={item['n_seeds']}){r2_text}")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8, loc="best")
    for j in range(len(variants), rows * cols):
        axes.flat[j].set_visible(False)
    fig.supxlabel("Test sample")
    fig.supylabel("Load")
    fig.suptitle(f"Test-set forecasts by model (first {n} samples)", y=1.01)
    _save(fig, output_path or results_dir / "comparison_predictions.png")


def plot_metrics_comparison(results_dir, metrics=("MAE", "RMSE", "R2"), output_path=None):
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "metrics_by_seed.csv")
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        grouped = df.groupby("model")[metric]
        means = grouped.mean().reindex(models)
        stds = grouped.std().reindex(models).fillna(0)
        ax.bar(np.arange(len(models)), means, yerr=stds, capsize=3,
               color=MODEL_COLORS[:len(models)], alpha=0.85)
        ax.set_xticks(np.arange(len(models)), models, rotation=25, ha="right")
        ax.set_title(f"{metric} (mean ± std)")
        ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path or results_dir / "comparison_metrics.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="outputs/dsr/uci_d")
    parser.add_argument("--n-points", type=int, default=300)
    args = parser.parse_args()
    results = Path(args.results_dir)
    plot_training_history(results / "training_history.csv")
    plot_prediction_comparison(results, n_points=args.n_points)
    plot_metrics_comparison(results)
    print(f"Saved figures to: {results.resolve()}")


if __name__ == "__main__":
    main()
