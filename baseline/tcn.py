from IPython import embed
import torch
from torch import nn
from torch.nn import functional as F

class Causal1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.conv1 = Causal1d(in_ch, out_ch, k, d)
        self.conv2 = Causal1d(out_ch, out_ch, k, d)
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        r = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.res:
            x = self.res(r) + x
        return self.relu(x)

class TCN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(in_ch, 16, 3, 1),
            TCNBlock(16, 16, 3, 2),
            TCNBlock(16, 16, 3, 4)
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, in_ch) -> (batch_size, in_ch, seq_len)
        x = x[:, :, None].permute(0, 2, 1)

        # x: (batch_size, in_ch, seq_len) -> (batch_size, in_ch, seq_len) 
        x = self.net(x)
        
        # 取最后一个时间步的输出  x: (batch_size, in_ch, seq_len) -> (batch_size, in_ch)
        x = x[:, :, -1] 
        
        # x: (batch_size, in_ch) -> (batch_size, 1)
        return self.fc(x)
        