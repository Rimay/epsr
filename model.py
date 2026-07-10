import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from IPython import embed


# -----------------   自适应发现数据周期 ---------------------
class adaptive_period_finder:
    def __init__(self, top_k=2):
        self.top_k = top_k

    def __call__(self, x):
        x = x.detach().cpu().numpy()
        x = x.mean(axis=0).squeeze()
        fft = np.fft.rfft(x)
        power = np.abs(fft)
        power[0] = 0
        
        idx = np.argsort(power)[-self.top_k:]
        periods = []
        N = len(x)
        for i in idx:
            if i!= 0:
                p = int(N / i)
                if p > 1:
                    periods.append(p)
        return sorted(list(set(periods)))

# -----------------   Trend-seasonal-residual 分解 ---------------------
# 移动平均趋势分解
# 输入 (B, C, T) 
# 输出 (B, C, T)
class trend_block(nn.Module):
    def __init__(self, k=25):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        return self.pool(x)

# 季节性分解
# 输入 (B, C, T) 
# 输出 (B, C, T)
class seasonal_block(nn.Module):
    def __init__(self, periods=[1, 7, 365], hidden=32):
        super().__init__()
        self.convs = nn.ModuleList()
        for p in periods:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(1, hidden, kernel_size=3, dilation=p, padding=p),
                    nn.ReLU()
                )
            )
        self.fusion = nn.Conv1d(len(periods) * hidden, 1, kernel_size=1)

    def forward(self, x):
        feats = []
        for conv in self.convs:
            feats.append(conv(x))
        out = torch.cat(feats, dim=1)
        out = self.fusion(out)
        return out

# 自适应周期(Seasonal)、趋势(Trend)、残差(Residual)分解
class APTR_decomposition(nn.Module):
    def __init__(self, periods=[1, 7, 365]):
        super().__init__()
        self.trend = trend_block(k=25)
        self.seasonal = seasonal_block(periods=periods)

    # x (B, C, T)
    def forward(self, x):
        trend = self.trend(x)
        seasonal = self.seasonal(x)
        residual = x - trend - seasonal
        return (trend, seasonal, residual)

# 周期融合
class period_fusion(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, feats):
        weights = []
        for f in feats:
            w = self.weight(f.mean(dim=-1))
            weights.append(w)
        weights = torch.softmax(torch.cat(weights, dim=1), dim=1)
        out = 0
        for i, f in enumerate(feats):
            w = weights[:, i:i+1]
            out += f * w.unsqueeze(-1)
        return out

# -----------------   temporal encoder (tcn / transformer / ...) ---------------------
class tcn_layer(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# S/T/R 时序编码器
# 输入 (B, C, T) 
# 输出 (B, 32, T)
class temporal_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            tcn_layer(1, 32, 1),
            tcn_layer(32, 32, 4),
            tcn_layer(32, 32, 8)
        )    
    def forward(self, x):
        return self.encoder(x)
    
# -----------------  prediction head  ---------------------
# 输入 (B, 32, T) 
# 输出 (B, 1, T)
class prediction_head(nn.Module):
    def __init__(self, hidden=32, output_len=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_len)
        )
    def forward(self, x):
        out = self.fc(x)
        return out

# -----------------  完整 Net  ---------------------
class net(nn.Module):
    def __init__(self, periods=[1, 7, 365], output_len=1):
        super().__init__()
        self.decomposition = APTR_decomposition(periods=periods)
        self.trend_encoder = temporal_encoder()
        self.seasonal_encoder = temporal_encoder()
        self.residual_encoder = temporal_encoder()
        self.fusion = period_fusion(dim=32)
        
        self.head = prediction_head(hidden=32, output_len=output_len)

    def forward(self, x):
        x = x[:, :, None].permute(0, 2, 1)

        trend, seasonal, residual = self.decomposition(x)
        h1 = self.trend_encoder(trend)
        h2 = self.seasonal_encoder(seasonal)
        h3 = self.residual_encoder(residual)
        h = self.fusion([h1, h2, h3])
        h = h[:, :, -1]

        y = self.head(h)
        return y


