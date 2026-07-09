import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

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


