"""Lag-residual TCN: autoregressive lag profile and optional known-future calendar branch."""
import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, cin, hidden, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(cin, hidden, 3, dilation)
        self.c2 = CausalConv1d(hidden, hidden, 3, dilation)
        self.n1, self.n2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.skip = nn.Conv1d(cin, hidden, 1) if cin != hidden else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.n1(self.c1(x))))
        h = self.drop(F.gelu(self.n2(self.c2(h))))
        return F.gelu(h + self.skip(x))


class TCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            TemporalBlock(in_dim, hidden, 1, dropout),
            TemporalBlock(hidden, hidden, 2, dropout),
            TemporalBlock(hidden, hidden, 4, dropout))

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]


class DSRNet(nn.Module):
    """Compatibility name; state labels are no longer model inputs."""
    VARIANTS = {"tcn", "lag_tcn", "calendar_lag_tcn"}

    def __init__(self, in_dim, state_indices=None, variant="lag_tcn",
                 hidden=32, state_hidden=16, dropout=0.2, num_states=3,
                 future_dim=0):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.encoder = TCNEncoder(in_dim, hidden, dropout)
        if variant == "tcn":
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))
            return

        # Six lag taps + two pooled history statistics = fixed 8-D profile.
        self.lag_encoder = nn.Sequential(nn.Linear(8, hidden), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.future_encoder = (nn.Sequential(nn.Linear(future_dim, hidden // 2), nn.GELU())
                               if variant == "calendar_lag_tcn" and future_dim > 0 else None)
        fused = hidden * 2
        if self.future_encoder is not None:
            fused += hidden // 2
        self.head = nn.Sequential(nn.Linear(fused, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
                                  nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x, future_cov=None):
        temporal = self.encoder(x)
        if self.variant == "tcn":
            return {"pred": self.head(temporal)}
        load = x[:, :, 0]
        # Daily lags for 28-step windows; hourly analogues for 168-step windows.
        if x.size(1) >= 168:
            offsets = (1, 2, 24, 48, 72, 168)
        else:
            offsets = (1, 2, 7, 14, 21, 28)
        taps = [load[:, -k:-k+1] if k > 1 else load[:, -1:] for k in offsets]
        lag_profile = torch.cat(taps + [F.adaptive_avg_pool1d(load.unsqueeze(1), 2).squeeze(1)], dim=1)
        parts = [temporal, self.lag_encoder(lag_profile)]
        if self.future_encoder is not None:
            if future_cov is None:
                raise ValueError("future_cov is required for calendar_lag_tcn")
            parts.append(self.future_encoder(future_cov))
        correction = self.head(torch.cat(parts, dim=1))
        return {"pred": load[:, -1:] + correction}
