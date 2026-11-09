import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=2, dilation=1):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.c2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        t = self.drop(self.act(self.c1(x)))
        t = self.drop(self.act(self.c2(t)))
        return self.act(t + self.skip(x))


class RASATTCN(nn.Module):
    def __init__(self, in_dim=8, hidden=16, state_dim=16, periodic_dim=16,
                 kernel=2, dilations=(1, 2), dropout=0.1, variant="full"):
        super().__init__()
        if variant not in {"full", "no_reliability", "no_gate", "no_periodic"}:
            raise ValueError(f"Unknown RASAT variant: {variant}")
        self.variant = variant
        blocks, c = [], in_dim
        for d in dilations:
            blocks.append(TCNBlock(c, hidden, kernel, d, dropout))
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.state_encoder = nn.Sequential(nn.Linear(hidden, 16), nn.GELU(), nn.Linear(16, state_dim))
        self.periodic_encoder = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, periodic_dim))
        self.reliability_net = nn.Sequential(nn.Linear(state_dim + 4, 16), nn.GELU(), nn.Linear(16, 1))
        self.base_head = nn.Sequential(nn.Linear(hidden + state_dim + periodic_dim, 16), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(16, 1))
        cd = hidden + state_dim + 2
        self.gate_head = nn.Sequential(nn.Linear(cd, 16), nn.GELU(), nn.Linear(16, 1))
        self.correction_head = nn.Sequential(nn.Linear(cd, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x, p, meta):
        p = p[:, -1, :]
        meta = meta[:, -1, :]
        h = self.tcn(x.transpose(1, 2))[:, :, -1]
        state = self.state_encoder(h)
        p_emb = self.periodic_encoder(p)

        r_raw, discrepancy = meta[:, 0:1], meta[:, 1:2]
        similarity, volatility = meta[:, 2:3], meta[:, 3:4]
        r_in = torch.cat([r_raw, discrepancy, similarity, state, volatility], dim=-1)
        reliability = 2 * torch.sigmoid(self.reliability_net(r_in))
        if self.variant == "no_reliability":
            reliability = torch.ones_like(reliability)

        p_adapt = reliability * p_emb
        if self.variant == "no_periodic":
            p_adapt = torch.zeros_like(p_adapt)

        y_base = self.base_head(torch.cat([h, state, p_adapt], dim=-1))
        corr_in = torch.cat([h, state, reliability, y_base], dim=-1)
        gate = torch.sigmoid(self.gate_head(corr_in))
        delta = self.correction_head(corr_in)
        pred = y_base if self.variant == "no_gate" else y_base + gate * delta

        return {"pred": pred, "y_base": y_base, "S_t": state,
                "R_t": reliability, "G_t": gate, "delta": delta}
