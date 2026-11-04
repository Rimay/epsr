import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 可视化特征随时间变化趋势
def plot_features(df):
    n_features = len(df.columns)
    n_rows = (n_features + 1) // 2  
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8 * n_rows))
    axes = axes.flatten()
    
    for i, column in enumerate(df.columns):
        ax = axes[i]
        ax.plot(df.index, df[column])
        ax.set_title(column, pad=15)
        ax.set_xlabel('')
        ax.grid(True)
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelsize=8)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98, hspace=0.5, wspace=0.2)
    plt.show()

# 可视化特征之间的相关性
def feature_analysis(df):
    corr = df.corr()
    print(corr)
    plt.subplots(figsize=(15, 12))
    sns.heatmap(corr, vmax=0.9, cmap='Blues', square=True, annot=True)
    plt.show()


# 可视化训练损失曲线
def plot_loss_curve(train_losses, title="Training Loss"):
    if isinstance(train_losses, dict):
        plt.figure(figsize=(10, 6))
        for name, losses in train_losses.items():
            plt.plot(losses, label=name)
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# 可视化模型预测结果
def plot_prediction(gt, preds):
    # gt 可能是 dict（{模型名: 真实值}）或单个数组；
    # 所有模型共用同一份真实值，取第一个即可
    true = np.asarray(next(iter(gt.values())) if isinstance(gt, dict) else gt).reshape(-1)

    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True Value", linewidth=1.5, color='green')
    
    if isinstance(preds, dict):
        for name, pred in preds.items():
            plt.plot(np.asarray(pred).reshape(-1), label=f"{name} Prediction", linewidth=1.5)
    else:
        plt.plot(np.asarray(preds).reshape(-1), label="Prediction", linewidth=1.5)    
    
    plt.title("Prediction vs Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power")
    plt.legend()
    plt.grid(True)
    plt.show()

def eval(test, preds):
    mae = {}
    mse = {}
    rmse = {}
    r2 = {}
    for name, pred in preds.items():
        # test 可能是 dict（{模型名: 真实值}）或单个数组
        true = np.asarray(test[name] if isinstance(test, dict) else test).reshape(-1)
        p = np.asarray(pred).reshape(-1)
        print(f"【Model】: {name}")
        mae[name] = mean_absolute_error(true, p)
        mse[name] = mean_squared_error(true, p)
        rmse[name] = np.sqrt(mse[name])
        r2[name] = r2_score(true, p) 
        print(f"MAE: {mae[name]:.4f}")
        print(f"MSE: {mse[name]:.4f}")
        print(f"RMSE: {rmse[name]:.4f}")
        print(f"R²: {r2[name]:.4f}")