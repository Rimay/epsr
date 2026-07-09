import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from IPython import embed


# 找到数据的周期
def find_period(data, top_k=2):
    fft = np.fft.rfft(data)
    power = np.abs(fft)
    power[0] = 0
    
    idx = np.argsort(power)[-top_k:]
    periods = []
    n = len(data)
    for i in idx:
        p = int(n / i)
        if p > 1:
            periods.append(p)
    periods = sorted(list(set(periods)))
    # embed()
    return periods

class Casual1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class mp_block(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, periods=[1, 7,365]):
        super().__init__()
        self.convs = nn.ModuleList()
        for p in periods:
            self.convs.append(Casual1d(in_ch, out_ch, k, d=p))
        self.fusion = nn.Conv1d(len(periods) * out_ch, out_ch, 1)        

    def forward(self, x):
        feats = []
        for conv in self.convs:
            feats.append(torch.relu(conv(x)))
        x = torch.cat(feats, dim=1)
        x = self.fusion(x)
        return x
        
class cycle_residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, fx):
        out = self.a * x
        seq_len = x.shape[-1]
        if seq_len > 7:
            x1 = F.pad(x[..., :-7], (7,0))
            out = out + self.b * x1
        if seq_len > 365:
            x2 = F.pad(x[..., :-365], (365,0))
            out = out + self.c * x2
        return out + fx

class apcr_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = mp_block(in_ch, out_ch, k=3, periods=[1, 7,365])
        self.cycle = cycle_residual()
        self.relu = nn.ReLU()
        self.down = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None)
    
    def forward(self, x):
        res = x
        if self.down:
            res = self.down(res)
        fx = self.mp(x)
        x = self.cycle(res, fx)
        return self.relu(x)
    
class APCR_TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            apcr_block(1, 16),
            apcr_block(16, 16),
            apcr_block(16, 16)
        )
        self.head = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x[:, :, None].permute(0, 2, 1)
        x = self.blocks(x)
        x = x[:, :, -1]
        return self.head(x)