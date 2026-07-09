from IPython import embed
import torch
from torch import nn
from torch.nn import functional as F

class DelayConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, delays=[0,1,3], bias=True):
        super().__init__()
        self.delays = delays
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, len(delays)))
        self.bias = nn.Parameter(torch.randn(out_ch)) if bias else None

    def forward(self, x):
        batch_size, in_ch, seq_len = x.shape
        max_delay = max(self.delays)

        x = F.pad(x, (max_delay, 0))
        xs = []
        for d in self.delays:
            xs.append(x[:, :, max_delay-d : max_delay-d+seq_len])
        xs = torch.stack(xs, dim=-1)
        y = torch.einsum("bctk, ock->bot", xs, self.weight)
        if self.bias is not None:
            y += self.bias.view(1, -1, 1)
        return y

class DelayBlock(nn.Module):
    def __init__(self, in_ch, out_ch, delays=[0,1,3]):
        super().__init__()
        self.conv = DelayConv1d(in_ch, out_ch, delays)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = out + identity
        return self.relu(out)

class DelayNet(nn.Module):
    def __init__(self, input_dim=1, hidden=32, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            DelayBlock(input_dim, hidden, delays=[0,1,3]),
            DelayBlock(hidden, hidden, delays=[0,2,6]),
            DelayBlock(hidden, hidden, delays=[0,4,12])
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, in_ch) -> (batch_size, in_ch, seq_len)
        x = x[:, :, None].permute(0, 2, 1)
        x = self.encoder(x)
        out = self.head(x)
        return out
