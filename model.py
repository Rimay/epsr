import torch
from torch import nn
import torch.nn.functional as F
from IPython import embed

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
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
    def __init__(self, in_dim=8, hidden=32, state_dim=16, periodic_dim=16,
            kernel=3, dilations=(1,2,4,8), dropout=0.1, variant="full"):
        super().__init__()
        self.variant = variant
        blocks, c = [], in_dim
        for d in dilations:
            blocks.append(TCNBlock(c, hidden, kernel, d, dropout))
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.state_encoder = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, state_dim))
        self.periodic_encoder = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, periodic_dim))
        self.reliability_net = nn.Sequential(nn.Linear(state_dim+4, 32), nn.GELU(), nn.Linear(32, 1))
        self.base_head = nn.Sequential(nn.Linear(hidden+state_dim+periodic_dim, 32), nn.GELU(), 
                                        nn.Dropout(dropout), nn.Linear(32, 1))
        cd = hidden + state_dim + 2
        self.gate_head = nn.Sequential(nn.Linear(cd, 32), nn.GELU(), nn.Linear(32, 1))
        self.correction_head = nn.Sequential(nn.Linear(cd, 32), nn.GELU(), nn.Linear(32, 1))

    # x [B, L, in_dim] (L=168 小时级 / 7 天级), p [B, 1, 2], meta [B, 1, 4]
    def forward(self, x, p, meta):
        # ---- 入口降维：pred_len=1，取预测步（最后一步）压成 2D，后续全在 [B, D] 上计算 ----
        p = p[:, -1, :]          # [B, 1, 2] -> [B, 2]
        meta = meta[:, -1, :]    # [B, 1, 4] -> [B, 4]

        # ---- TCN 主干 ----
        H_t = self.tcn(x.transpose(1, 2))[:, :, -1]   # transpose [B,8,168] -> conv [B,32,168] -> 取末时刻 [B, 32]
        S_t = self.state_encoder(H_t)                 # [B, 32] -> [B, 16]
        p_emb = self.periodic_encoder(p)              # [B, 2]  -> [B, 16]

        # ---- 可靠性描述符：按列切片，保持 [B, 1]（不能用 meta[:, 0]，那会得到 [B]）----
        r_raw       = meta[:, 0:1]   # [B, 1]
        discrepancy = meta[:, 1:2]   # [B, 1]
        similarity  = meta[:, 2:3]   # [B, 1]
        volatility  = meta[:, 3:4]   # [B, 1]

        # ---- 可靠性标量 R_t ----
        r_in = torch.cat([r_raw, discrepancy, similarity, S_t, volatility], dim=-1)   # [B, 16+4] = [B, 20]
        # 2*sigmoid 使 R_t 以 1 为中心（(0,2)），避免恒小于 1 时对周期嵌入的乘性衰减
        R_t = 2 * torch.sigmoid(self.reliability_net(r_in))                          # [B, 1]

        # ---- 周期嵌入按可靠性自适应缩放：广播 [B,1] * [B,16] -> [B, 16] ----
        p_adapt = R_t * p_emb

        # ---- 基线预测头 ----
        y_base = self.base_head(torch.cat([H_t, S_t, p_adapt], dim=-1))   # cat [B, 32+16+16] = [B, 64] -> [B, 1]

        # ---- 门控修正 ----
        corr_in = torch.cat([H_t, S_t, R_t, y_base], dim=-1)   # [B, 32+16+1+1] = [B, 50]，与 cd=50 对齐
        G_t = torch.sigmoid(self.gate_head(corr_in))           # [B, 1]，门控系数
        delta = self.correction_head(corr_in)                  # [B, 1]，修正量以 corr_in 为输入（原 bug：误喂 G_t）
        pred = y_base + G_t * delta                            # [B, 1]

        return {"pred": pred, "y_base": y_base, "S_t": S_t, "R_t": R_t, "G_t": G_t, "delta": delta}