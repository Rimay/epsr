"""DSR-Net 数据管线：读数 → 特征构造 → 滑窗数据集 → 时间切分。

防泄露原则：所有统计量（均值/标准差/分位数阈值）只在训练段拟合；
WindowDataset 丢弃含 NaN 的窗口；chronological_split 严格按时间顺序切分。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

DATA_PATH = "data"


def load_df(dataset_name="uci", data_root=None):
    """读取原始负荷数据，返回 (分钟级原始序列, 天级均值, 小时级均值)。"""
    root = Path(data_root or DATA_PATH).expanduser()
    key = dataset_name.lower()

    if key == "uci":
        path = root / "uci" / "household_power_consumption.txt"
        if not path.exists():
            raise FileNotFoundError(f"UCI file not found: {path}")
        raw = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
        raw["timestamp"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str),
                                        format="%d/%m/%Y %H:%M:%S", errors="coerce")
        raw["load"] = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    elif key in {"gefcom2014", "gefcom"}:
        path = root / "GEFCom2014" / "Load" / "clean" / "gefcom2014_load_merged.csv"
        if not path.exists():
            raise FileNotFoundError(f"GEFCom2014 merged file not found: {path}")
        raw = pd.read_csv(path, low_memory=False)
        if not {"timestamp", "load"}.issubset(raw.columns):
            raise ValueError("GEFCom2014 CSV must contain timestamp and load")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw["load"] = pd.to_numeric(raw["load"], errors="coerce")
    else:
        raise ValueError("dataset must be uci or GEFCom2014")

    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    # 同一时间戳多条记录取均值，统一为单调时间索引
    raw = raw.groupby("timestamp", as_index=False)["load"].mean().set_index("timestamp").sort_index()
    return raw, raw[["load"]].resample("D").mean(), raw[["load"]].resample("h").mean()


def make_frame(source, freq="d", train_fraction=0.70):
    """构造特征表，返回 (frame, features, meta)。

    frame 列：
        load           原始负荷
        delta / delta2 一、二阶差分
        rolling_std_*  波动率（h 用 24 点窗，d 用 7 点窗）
        *_sin / *_cos  日历周期特征
        state          负载状态标签（0=Normal / 1=Peak / 2=Ramp）
        target_z       标准化后的预测目标
    meta 记录切分边界、阈值与标准化统计量，供训练和反变换复用。
    """
    if freq not in {"h", "d"}:
        raise ValueError("freq must be h or d")
    frame = source[["load"]].copy().sort_index()
    frame["load"] = pd.to_numeric(frame["load"], errors="coerce")

    # ---- 时间切分边界：70/15/15 ----
    n = len(frame)
    train_end = int(n * train_fraction)
    val_end = int(n * 0.85)
    if train_end < 100 or val_end <= train_end or n - val_end < 20:
        raise ValueError(f"Insufficient rows ({n}) for chronological 70/15/15 split")

    # ---- 按频率选择日历特征与波动率窗口 ----
    if freq == "h":
        window = 24
        frame["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
        frame["dow_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
        volatility = "rolling_std_24"
        calendar = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        window = 7
        frame["dow_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
        frame["month_sin"] = np.sin(2 * np.pi * (frame.index.month - 1) / 12)
        frame["month_cos"] = np.cos(2 * np.pi * (frame.index.month - 1) / 12)
        volatility = "rolling_std_7"
        calendar = ["dow_sin", "dow_cos", "month_sin", "month_cos"]

    frame["delta"] = frame["load"].diff()
    frame["delta2"] = frame["delta"].diff()
    frame[volatility] = frame["load"].rolling(window, min_periods=window).std()
    features = ["load", "delta", "delta2", volatility] + calendar

    # ---- 标准化统计量与状态阈值：只用训练段拟合，防止泄露 ----
    train_part = frame.iloc[:train_end]
    peak = float(train_part["load"].quantile(0.90))
    ramp = float(train_part["delta"].abs().quantile(0.90))
    y_mean = float(train_part["load"].mean())
    y_std = float(train_part["load"].std())
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0

    # ---- 状态标签：先标 Ramp（大幅变化），再用 Peak（高负荷）覆盖，同时满足时归为 Peak ----
    states = np.zeros(n, dtype=np.int64)
    states[np.abs(frame["delta"].to_numpy()) >= ramp] = 2
    states[frame["load"].to_numpy() >= peak] = 1
    frame["state"] = states

    # ---- 标准化特征与目标 ----
    frame["target_z"] = (frame["load"] - y_mean) / y_std
    x_mean = train_part[features].mean()
    x_std = train_part[features].std().replace(0, 1).fillna(1)
    frame[features] = (frame[features] - x_mean) / x_std
    frame = frame.replace([np.inf, -np.inf], np.nan)

    meta = {
        "train_end": train_end,
        "val_end": val_end,
        "peak_threshold": peak,
        "ramp_threshold": ramp,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_mean": x_mean.to_dict(),
        "feature_std": x_std.to_dict(),
        "state_names": ["Normal", "Peak", "Ramp"],\n        "calendar_features": calendar,
    }
    return frame, features, meta


class WindowDataset(Dataset):
    """滑窗数据集：用历史 [t-L, t) 预测 t 时刻的目标与状态。

    属性：
        x              [N, L, F] 标准化特征窗口
        y              [N, 1]    标准化目标
        states         [N]       状态标签
        target_indices [N]       每个窗口对应的目标时刻 t（用于按时间切分）

    只保留特征窗口与目标均无 NaN 的样本。
    """

    def __init__(self, frame, features, seq_len=168, future_features=None):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        values = frame[list(features)].to_numpy(dtype=np.float32)
        self.future_features = list(future_features or [])
        future_values = (frame[self.future_features].to_numpy(dtype=np.float32)
                         if self.future_features else np.zeros((len(frame), 0), dtype=np.float32))
        targets = frame["target_z"].to_numpy(dtype=np.float32)
        states = frame["state"].to_numpy(dtype=np.int64)

        self.target_indices = []
        xs, ys, ss, fs = [], [], [], []
        for t in range(seq_len, len(frame)):
            window = values[t - seq_len:t]
            if np.isfinite(window).all() and np.isfinite(targets[t]):
                xs.append(window)
                ys.append([targets[t]])
                ss.append(states[t])
                fs.append(future_values[t])
                self.target_indices.append(t)
        if not xs:
            raise ValueError("No valid windows; check data and seq_len")

        self.x = torch.tensor(np.asarray(xs), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(ys), dtype=torch.float32)
        self.states = torch.tensor(np.asarray(ss), dtype=torch.long)
        self.future = torch.tensor(np.asarray(fs), dtype=torch.float32)
        self.target_indices = np.asarray(self.target_indices, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.states[i], self.future[i]


def chronological_split(dataset, train_end, val_end):
    """按窗口目标时刻 t 与切分边界，划分训练/验证/测试三个 Subset。"""
    t = dataset.target_indices
    groups = [
        np.flatnonzero(t < train_end),
        np.flatnonzero((t >= train_end) & (t < val_end)),
        np.flatnonzero(t >= val_end),
    ]
    if any(len(g) == 0 for g in groups):
        raise ValueError("Empty chronological partition")
    return tuple(Subset(dataset, g.tolist()) for g in groups)
