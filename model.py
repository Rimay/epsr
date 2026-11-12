"""DSR-Net: Dynamic State-aware Routing Network（动态状态感知路由网络）。

结构总览（输入历史窗口 x，形状 [B, L, F]）：
    1. TCNEncoder 提取时序表征 h [B, hidden]；
    2. 非 tcn 变体：取末时刻的状态描述符经 state_encoder 编码后与 h 拼接；
    3. forecast_head 输出预测 [B, 1]；
    4. 含状态分类的变体额外输出 state_logits/state_probs；
    5. 含专家路由的变体（full/no_transition）：每个负载状态一个专家头，
       用 state_probs 对各专家预测加权融合。

设计约束：推理时绝不使用未来信息；状态描述符只取自历史窗口。
"""
import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果一维卷积：只看当前时刻及更早，不泄露未来。"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation  # 仅左侧补零，保持时间长度不变
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    """TCN 基本块：两层因果卷积 + BatchNorm + GELU + 残差连接。"""
    def __init__(self, in_channels, hidden, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, hidden, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=3, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)
        self.skip = nn.Conv1d(in_channels, hidden, 1) if in_channels != hidden else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        h = self.dropout(F.gelu(self.norm2(self.conv2(h))))
        return F.gelu(h + self.skip(x))


class TCNEncoder(nn.Module):
    """三层膨胀因果卷积（dilation=1/2/4），取最后时刻的表征。"""
    def __init__(self, in_channels, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            TemporalBlock(in_channels, hidden, dilation=1, dropout=dropout),
            TemporalBlock(hidden, hidden, dilation=2, dropout=dropout),
            TemporalBlock(hidden, hidden, dilation=4, dropout=dropout),
        )

    def forward(self, x):
        """x: [B, L, F] → [B, hidden]（先转成通道优先，再取末时刻）。"""
        return self.net(x.transpose(1, 2))[:, :, -1]


class MLP(nn.Module):
    """两层隐层 + 线性输出层的通用 MLP 头。"""
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
    """DSR-Net 主模型。
    变体说明（VARIANTS）：
        tcn            纯 TCN 基线：只喂 load 单通道，无状态/路由分支
        state          TCN + 状态描述符拼接
        state_aux      state + 状态分类辅助任务（计入 state 损失）
        full           state_aux + 专家路由（计入 expert 损失）
        no_transition  与 full 同结构，但训练时不计入 state 损失（用于消融）
    """

    VARIANTS = {"tcn", "state", "state_aux", "full", "no_transition"}

    def __init__(self, in_dim, state_indices, variant="full",
                 hidden=32, state_hidden=16, dropout=0.2, num_states=3):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.state_indices = tuple(state_indices)

        self.temporal = TCNEncoder(in_dim, hidden, dropout)
        if variant == "tcn":
            fused_dim = hidden
        else:
            self.state_encoder = MLP(len(state_indices), state_hidden, state_hidden, dropout)
            fused_dim = hidden + state_hidden

        self.forecast_head = MLP(fused_dim, hidden, 1, dropout)
        self.state_classifier = (MLP(fused_dim, hidden, num_states, dropout)
                                 if variant in {"state_aux", "full", "no_transition"} else None)
        self.experts = (nn.ModuleList(MLP(fused_dim, hidden, 1, dropout) for _ in range(num_states))
                        if variant in {"full", "no_transition"} else None)

    def forward(self, x):
        """x: [B, L, F]（tcn 变体只传 load 单通道，见 main.select_inputs）。

        返回 dict：
            pred            [B, 1]  最终预测
            representation  [B, fused_dim] 融合表征
            state_logits / state_probs  仅状态分类变体返回
            expert_preds    [B, num_states] 仅路由变体返回
        """
        feat = self.temporal(x)  # [B, hidden]
        if self.variant != "tcn":
            state_feat = self.state_encoder(x[:, -1, list(self.state_indices)])  # 仅末时刻描述符
            feat = torch.cat([feat, state_feat], dim=-1)

        logits = self.state_classifier(feat) if self.state_classifier is not None else None
        probs = torch.softmax(logits, dim=-1) if logits is not None else None

        if self.experts is not None:
            expert = torch.cat([head(feat) for head in self.experts], dim=1)  # [B, num_states]
            pred = (expert * probs).sum(dim=1, keepdim=True)  # 按状态概率加权融合
        else:
            expert = None
            pred = self.forecast_head(feat)

        out = {"pred": pred, "representation": feat}
        if logits is not None:
            out.update(state_logits=logits, state_probs=probs)
        if expert is not None:
            out["expert_preds"] = expert
        return out


