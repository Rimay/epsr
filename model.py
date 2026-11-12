"""DSR-Net v2: Multi-scale State-aware Routing Network.

The ordinary TCN baseline is intentionally retained as a conventional reference.
DSR variants add:
  1) multi-scale causal temporal encoding (parallel dilation paths),
  2) attention pooling over the history window,
  3) a compact autoregressive/seasonal history branch,
  4) gated fusion with explicit dynamic-state descriptors,
  5) state-aware expert routing (Full / No_transition).

Only historical inputs enter forward(); target-time state labels are supervision
and evaluation labels only.
"""
import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    """Conventional residual TCN block used by the baseline."""
    def __init__(self, in_channels, hidden, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, hidden, 3, dilation)
        self.conv2 = CausalConv1d(hidden, hidden, 3, dilation)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)
        self.skip = nn.Conv1d(in_channels, hidden, 1) if in_channels != hidden else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        h = self.dropout(F.gelu(self.norm2(self.conv2(h))))
        return F.gelu(h + self.skip(x))


class TCNEncoder(nn.Module):
    """Plain TCN baseline; do not silently upgrade the baseline itself."""
    def __init__(self, in_channels, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            TemporalBlock(in_channels, hidden, 1, dropout),
            TemporalBlock(hidden, hidden, 2, dropout),
            TemporalBlock(hidden, hidden, 4, dropout),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]


class MultiScaleBlock(nn.Module):
    """Parallel causal receptive fields followed by learned channel mixing."""
    def __init__(self, in_channels, hidden, dropout):
        super().__init__()
        branch = max(8, hidden // 2)
        self.paths = nn.ModuleList([
            nn.Sequential(CausalConv1d(in_channels, branch, 3, d),
                          nn.BatchNorm1d(branch), nn.GELU())
            for d in (1, 2, 4)
        ])
        self.merge = nn.Conv1d(branch * 3, hidden, 1)
        self.norm = nn.BatchNorm1d(hidden)
        self.skip = nn.Conv1d(in_channels, hidden, 1) if in_channels != hidden else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        multi = torch.cat([path(x) for path in self.paths], dim=1)
        h = self.dropout(F.gelu(self.norm(self.merge(multi))))
        return F.gelu(h + self.skip(x))


class AttentionPool(nn.Module):
    """Content-based temporal pooling; all positions are historical."""
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden // 2),
                                   nn.Tanh(), nn.Linear(hidden // 2, 1))

    def forward(self, sequence):
        # sequence: [B, L, H]
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden=32, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            MultiScaleBlock(in_channels, hidden, dropout),
            MultiScaleBlock(hidden, hidden, dropout),
            MultiScaleBlock(hidden, hidden, dropout),
        )
        self.pool = AttentionPool(hidden)
        self.norm = nn.LayerNorm(hidden * 2)

    def forward(self, x):
        seq = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        # Last-state representation preserves recency; attention captures context.
        return self.norm(torch.cat([seq[:, -1], self.pool(seq)], dim=-1))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DSRNet(nn.Module):
    """DSR-Net v2 with a fixed interface compatible with the experiment runner."""
    VARIANTS = {"tcn", "state", "state_aux", "full", "no_transition"}

    def __init__(self, in_dim, state_indices, variant="full",
                 hidden=32, state_hidden=16, dropout=0.2, num_states=3):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.state_indices = tuple(state_indices)

        if variant == "tcn":
            self.temporal = TCNEncoder(in_dim, hidden, dropout)
            fused_dim = hidden
        else:
            # DSR encoder uses multiple receptive fields and temporal attention.
            self.temporal = MultiScaleTemporalEncoder(in_dim, hidden, dropout)
            # A short raw-load path retains recent level/weekly-shape information.
            self.history_encoder = MLP(8, hidden, hidden * 2, dropout)
            self.history_gate = nn.Sequential(nn.Linear(hidden * 4, hidden * 2),
                                              nn.Sigmoid())
            self.state_encoder = MLP(len(state_indices), state_hidden,
                                     state_hidden, dropout)
            fused_dim = hidden * 2 + state_hidden

        self.forecast_head = MLP(fused_dim, hidden, 1, dropout)
        self.state_classifier = (
            MLP(fused_dim, hidden, num_states, dropout)
            if variant in {"state_aux", "full", "no_transition"} else None
        )
        self.experts = (
            nn.ModuleList([MLP(fused_dim, hidden, 1, dropout)
                           for _ in range(num_states)])
            if variant in {"full", "no_transition"} else None
        )

    def forward(self, x):
        # x: [B, L, F]; no future labels or future observations are consumed.
        if self.variant == "tcn":
            feat = self.temporal(x)
        else:
            temporal_feat = self.temporal(x)
            # Adaptive pooling makes this branch work for both daily and hourly windows.
            load_history = F.adaptive_avg_pool1d(
                x[:, :, :1].transpose(1, 2), 8
            ).squeeze(1)
            history_feat = self.history_encoder(load_history)
            gate = self.history_gate(torch.cat([temporal_feat, history_feat], dim=-1))
            fused_temporal = gate * temporal_feat + (1.0 - gate) * history_feat
            state_feat = self.state_encoder(x[:, -1, list(self.state_indices)])
            feat = torch.cat([fused_temporal, state_feat], dim=-1)

        logits = self.state_classifier(feat) if self.state_classifier is not None else None
        probs = torch.softmax(logits, dim=-1) if logits is not None else None

        if self.experts is not None:
            expert = torch.cat([head(feat) for head in self.experts], dim=1)
            pred = torch.sum(expert * probs, dim=1, keepdim=True)
        else:
            expert = None
            pred = self.forecast_head(feat)

        out = {"pred": pred, "representation": feat}
        if logits is not None:
            out.update(state_logits=logits, state_probs=probs)
        if expert is not None:
            out["expert_preds"] = expert
        return out
