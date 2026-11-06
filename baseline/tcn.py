import torch
from torch import nn
from torch.nn import functional as F


class Causal1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.conv1 = Causal1d(in_ch, out_ch, k, d)
        self.conv2 = Causal1d(out_ch, out_ch, k, d)
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.res is not None:
            x = self.res(residual) + x
        return self.relu(x)


class TCN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            TCNBlock(in_ch, 16, 3, 1),
            TCNBlock(16, 16, 3, 2),
            TCNBlock(16, 16, 3, 4)
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # 输入可以是 [B,L] 或 [B,L,C]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2).contiguous()
        x = self.net(x)
        return self.fc(x[:, :, -1])
