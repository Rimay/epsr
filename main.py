import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython import embed

from vis import plot_features, feature_analysis, plot_loss_curve, plot_prediction, eval
from dataset import load_df, window_dataset, make_frame, split_dataset
from model import RASATTCN
from baseline.ml_model import arima_model, xgb_model
from baseline.rnn import GRU, LSTM
from baseline.tcn import TCN
from baseline.delaynet import DelayNet
from baseline.apcr_tcn import APCR_TCN


# ds_name = ['uci', 'GEFCom2014']
# ══ 预测粒度：'h'=小时级(seq_len=168)，'d'=天级(seq_len=7) ══
FREQ = 'd'
SEQ_LEN = {'h': 168, 'd': 7}[FREQ]      # 天级可试 7(一周) / 14(两周)
P_COLS  = {'h': ['lag24_z', 'lag168_z'], 'd': ['lag1_z', 'lag7_z']}[FREQ]
TRAIN_FRAC = 0.7                        # 训练段比例：标准化统计量与切分边界共用

# 统一前向调用：基线模型只用负荷序列（单通道 [B,L]），RASATTCN 额外需要周期 p 与 meta
def forward_model(model, x, p, meta):
    if isinstance(model, RASATTCN):
        return model(x, p, meta)['pred']
    return model(x[:, :, 0])

# 训练模型，ds：切分后的训练集子集
def train(ds, model):
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crt = nn.MSELoss()

    epochs, train_loss = 100, []
    for e in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, p, meta in loader:
            x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
            opt.zero_grad()
            pred = forward_model(model, x, p, meta)
            # y 是 [B,1,1]、pred 是 [B,1]，直接 MSE 会广播成 [B,B,1]，必须 squeeze
            loss = crt(pred, y.squeeze(-1))
            loss.backward()
            opt.step()  
            total_loss += loss.item()
        total_loss /= len(loader) 
        train_loss.append(total_loss)
        print(f'Epoch {e}, Loss: {total_loss:.4f}')         # 打印每个epoch的损失
    return model, train_loss

# 测试模型，ds：切分后的测试集子集
def test(ds, model):
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        true_list, pred_list = [], []
        for x, y, p, meta in loader:
            x, y, p, meta = x.to(device), y.to(device), p.to(device), meta.to(device)
            pred = forward_model(model, x, p, meta)
            true_list.append(y.cpu().numpy())
            pred_list.append(pred.cpu().numpy())
    gt = np.concatenate(true_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    return gt, pred

if __name__ == '__main__':
    print("Loading dataset...")
    df, dfd, dfh = load_df(ds_name='uci')
    src = dfh if FREQ == 'h' else dfd        # 天级用 dfd
    f, features, y_mean, y_std, thresh = make_frame(src, freq=FREQ, fit_frac=TRAIN_FRAC)

    # 按时间顺序切分，避免训练集混入未来样本
    ds = window_dataset(f, features, seq_len=SEQ_LEN, p_cols=P_COLS)
    tr_ds, va_ds, te_ds = split_dataset(ds, ratios=(TRAIN_FRAC, 0.15, 0.15))
    print(f"样本数 total={len(ds)}, train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")
    
    # 模型训练、测试、可视化
    models = {
        'TCN': TCN(),
        'APCR_TCN': APCR_TCN(),
        'DelayNet': DelayNet(),
        'GRU': GRU(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1),
        'LSTM': LSTM(input_dim=1, hidden_dim=64, num_layers=2, pred_len=1),
        "RASAT_TCN": RASATTCN(in_dim=len(features)) # in_dim 跟随特征数，避免硬编码 8 与实际列数不一致
    }
    train_losses, preds, gt = {}, {}, {}
    print("Training models...")
    # embed()
    for name, model in models.items():
        print(f"Training {name}...")
        model, train_losses[name] = train(tr_ds, model)
        print(f"Testing {name}...")
        gt[name], preds[name] = test(te_ds, model)
    
    print("Plotting loss curve...")
    plot_loss_curve(train_losses, title='Train Loss Comparison')    
    
    print("Plotting prediction...")
    plot_prediction(gt, preds)

    print("Evaluating models...")
    eval(gt, preds)

