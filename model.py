"""DSR-Net: Dynamic State-aware Routing Network for short-term load forecasting.

Inputs are historical load-derived features only. Future operating states are used
as training labels, never as inference-time routing inputs.
"""
from typing import Dict, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.skip = (nn.Conv1d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return F.gelu(y + self.skip(x))


class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32,
                 dilations: Sequence[int] = (1, 2, 4), dropout: float = 0.2):
        super().__init__()
        blocks = []
        current = in_channels
        for dilation in dilations:
            blocks.append(TemporalBlock(current, hidden, 3, dilation, dropout))
            current = hidden
        self.network = nn.Sequential(*blocks)
        self.output_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, features] -> final causal representation [batch, hidden]
        return self.network(x.transpose(1, 2))[:, :, -1]


class StateFeatureEncoder(nn.Module):
    """Encode explicit level, slope, curvature and rolling-volatility descriptors."""
    def __init__(self, state_dim: int, hidden: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU()
        )
        self.output_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSRNet(nn.Module):
    """Dynamic State-aware Routing Network.

    Variants:
      tcn: univariate temporal encoder + one regression head.
      state: temporal + explicit state features + one head.
      state_aux: state model + future-state classifier auxiliary task.
      full: state_aux + Normal/Peak/Ramp experts with soft probability routing.
    """
    VARIANTS = {"tcn", "state", "state_aux", "full"}

    def __init__(self, in_dim: int, state_indices: Sequence[int],
                 hidden: int = 32, state_hidden: int = 16,
                 dropout: float = 0.2, variant: str = "full",
                 num_states: int = 3):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        if not state_indices:
            raise ValueError("state_indices cannot be empty")
        if max(state_indices) >= in_dim or min(state_indices) < 0:
            raise ValueError("state_indices out of range")
        self.variant = variant
        self.state_indices = tuple(state_indices)
        self.encoder = TCNEncoder(in_dim, hidden, dropout=dropout)
        self.state_encoder = StateFeatureEncoder(len(state_indices), state_hidden, dropout)
        fused_dim = hidden if variant == "tcn" else hidden + state_hidden
        self.forecast_head = RegressionHead(fused_dim, hidden, dropout)
        self.state_classifier = (
            nn.Sequential(nn.Linear(fused_dim, hidden), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden, num_states))
            if variant in {"state_aux", "full"} else None
        )
        self.experts = (nn.ModuleList([
            RegressionHead(fused_dim, hidden, dropout) for _ in range(num_states)
        ]) if variant == "full" else None)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        temporal = self.encoder(x)
        if self.variant == "tcn":
            fused = temporal
        else:
            state_raw = x[:, -1, list(self.state_indices)]
            state_emb = self.state_encoder(state_raw)
            fused = torch.cat([temporal, state_emb], dim=-1)

        logits = self.state_classifier(fused) if self.state_classifier is not None else None
        probs = torch.softmax(logits, dim=-1) if logits is not None else None

        if self.variant == "full":
            expert_preds = torch.cat([head(fused) for head in self.experts], dim=1)
            pred = torch.sum(expert_preds * probs, dim=1, keepdim=True)
        else:
            pred = self.forecast_head(fused)
            expert_preds = None

        result = {"pred": pred, "representation": fused}
        if logits is not None:
            result["state_logits"] = logits
            result["state_probs"] = probs
        if expert_preds is not None:
            result["expert_preds"] = expert_preds
        return result


# Backward-compatible import name for notebooks/scripts that still import RASATTCN.
# The old forward(x, p, meta) signature is intentionally not retained: routing now
# depends only on model-predicted state probabilities, not hand-crafted meta inputs.
RASATTCN = DSRNet
