import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_analysis import plot_features, feature_analysis
import torch
from torch import nn
from IPython import embed

from dataset import uci_dataset
from baseline.ml_model import arima_model, xgb_model, eval
from baseline.rnn import GRU, LSTM
from baseline.tcn import TCN
from baseline.delaynet import DelayNet
from apcr_tcn import find_period, APCR_TCN


# 加载数据集
def dataset():
    # 加载数据集
    df = pd.read_csv('./data/household/household_power_consumption.txt', sep=';', header=0, low_memory=False,
        engine='c', na_values='?')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    # 数据预处理, 包括缺失值处理,下采样, 季节性分解
    df.dropna(inplace=True)
    dfh, dfd = df['Global_active_power'].resample('h').mean(),\
                 df['Global_active_power'].resample('d').mean()
    dfh.dropna(inplace=True)
    dfd.dropna(inplace=True)
    # dfh.plot()
    # dfd.plot()
    # plt.show()
    # decom = seasonal_decompose(dfd, model='multiplicative', period=12)
    # decom.plot()
    # plt.show()

    return df, dfd, dfh

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
    plt.figure(figsize=(12, 6))
    plt.plot(gt[:, 0], label="True Value", linewidth=1.5, color='green')
    
    if isinstance(preds, dict):
        for name, pred in preds.items():
            plt.plot(pred[:, 0], label=f"{name} Prediction", linewidth=1.5)
    else:
        plt.plot(preds[:, 0], label="Prediction", linewidth=1.5)    
    
    plt.title("Prediction vs Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power")
    plt.legend()
    plt.grid(True)
    plt.show()

# 训练模型
def train(data, model):
    dataset = uci_dataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crt = nn.MSELoss()

    epochs, train_loss = 100, []
    for e in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crt(pred, y)
            loss.backward()
            opt.step()  
            total_loss += loss.item()
        total_loss /= len(loader) 
        # 打印每个epoch的损失
        train_loss.append(total_loss)
        print(f'Epoch {e}, Loss: {total_loss:.4f}')
    return model, train_loss

# 测试模型
def test(data, model):
    dataset = uci_dataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        true_list, pred_list = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            true_list.append(y.cpu().numpy())
            pred_list.append(pred.cpu().numpy())
    gt = np.concatenate(true_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    return gt, pred


if __name__ == '__main__':
    # 加载数据集
    df, dfd, dfh = dataset()
    train_df, test_df = dfd[:'2009'], dfd['2010':]
    
    # 可视化特征随时间变化趋势
    # plot_features(df)
    # feature_analysis(df)
    
    # arima xgboost训练及测试
    # ari = arima_model(dfd)
    # xgb = xgb_model(dfd)

    # 找到数据的周期
    # periods = find_period(dfd)
    # print(periods)

    # 模型训练、测试、可视化
    models = {
        # 'TCN': TCN(),
        # 'APCR_TCN': APCR_TCN(),
        # 'DelayNet': DelayNet(),
        'GRU': GRU(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1),
        'LSTM': LSTM(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1)
    }
    train_losses = {}
    preds = {}
    for name, model in models.items():
        model, train_losses[name] = train(train_df, model)
        gt, preds[name] = test(test_df, model)
    plot_loss_curve(train_losses, title='Train Loss Comparison')    
    plot_prediction(gt, preds)
    eval(gt, preds)
