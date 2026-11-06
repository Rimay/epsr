import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_features(df):
    n_features = len(df.columns)
    n_rows = (n_features + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column])
        axes[i].set_title(column, pad=15)
        axes[i].grid(True)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def feature_analysis(df):
    corr = df.corr()
    print(corr)
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, vmax=0.9, cmap='Blues', square=True, annot=True)
    plt.show()


def plot_loss_curve(losses, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    for name, values in losses.items():
        plt.plot(values, label=name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction(gt, preds, y_mean=None, y_std=None):
    true = np.asarray(next(iter(gt.values())) if isinstance(gt, dict) else gt).reshape(-1)
    if y_mean is not None and y_std is not None:
        true = true * y_std + y_mean
    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True Value", linewidth=1.8, color="black")
    for name, pred in preds.items():
        values = np.asarray(pred).reshape(-1)
        if y_mean is not None and y_std is not None:
            values = values * y_std + y_mean
        plt.plot(values, label=f"{name} Prediction", linewidth=1.1)
    plt.title("Prediction vs Ground Truth")
    plt.xlabel("Test sample")
    plt.ylabel("Load (original unit)" if y_mean is not None else "Standardized Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def eval(test, preds, y_mean=None, y_std=None):
    results = {}
    for name, pred in preds.items():
        true = np.asarray(test[name] if isinstance(test, dict) else test).reshape(-1)
        p = np.asarray(pred).reshape(-1)
        if y_mean is not None and y_std is not None:
            true = true * y_std + y_mean
            p = p * y_std + y_mean
        mae = mean_absolute_error(true, p)
        mse = mean_squared_error(true, p)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, p)
        results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        print(f"【Model】: {name}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
    return results
