"""Leakage-aware time-series data preparation for UCI and GEFCom2014."""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

DATA_PATH = "data"


def _resolve_data_path(data_root: Optional[str] = None) -> Path:
    return Path(data_root or DATA_PATH).expanduser()


def load_df(ds_name: str = "uci", data_root: Optional[str] = None):
    root = _resolve_data_path(data_root)
    key = ds_name.lower()
    if key == "uci":
        path = root / "uci" / "household_power_consumption.txt"
        if not path.exists():
            raise FileNotFoundError(f"UCI data file not found: {path}")
        raw = pd.read_csv(path, sep=";", low_memory=False, na_values="?")
        raw["datetime"] = pd.to_datetime(
            raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S",
            errors="coerce"
        )
        raw = raw.dropna(subset=["datetime"])
        raw = raw.set_index("datetime").drop(columns=["Date", "Time"])
        raw = raw.rename(columns={"Global_active_power": "load"})
        raw["load"] = pd.to_numeric(raw["load"], errors="coerce")
    elif key in {"gefcom2014", "gefcom"}:
        path = root / "GEFCom2014" / "Load" / "clean" / "gefcom2014_load_merged.csv"
        if not path.exists():
            raise FileNotFoundError(f"GEFCom2014 merged load file not found: {path}")
        raw = pd.read_csv(path)
        if "timestamp" not in raw or "load" not in raw:
            raise ValueError("GEFCom2014 CSV must contain timestamp and load columns")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw["load"] = pd.to_numeric(raw["load"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"]).set_index("timestamp")
        raw = raw.drop(columns=["source_task"], errors="ignore")
    else:
        raise ValueError("ds_name must be 'uci' or 'GEFCom2014'")

    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="mean")]
    raw = raw.dropna(subset=["load"])
    hourly = raw["load"].resample("h").mean().to_frame()
    daily = raw["load"].resample("D").mean().to_frame()
    return raw, daily.asfreq("D"), hourly.asfreq("h")


def _safe_std(series: pd.Series) -> pd.Series:
    return series.std().replace(0, 1).fillna(1)


def make_frame(df: pd.DataFrame, freq: str = "h", fit_frac: float = 0.7):
    if not 0 < fit_frac < 1:
        raise ValueError("fit_frac must be between 0 and 1")
    if "load" not in df:
        raise ValueError("Input dataframe must contain a 'load' column")
    f = df[["load"]].copy().sort_index()
    if freq == "h":
        rolling_window, lag_short, lag_long = 24, 24, 168
        rolling_col, lag_short_col, lag_long_col = "rolling_std_24", "lag24", "lag168"
        f["hour_sin"] = np.sin(2 * np.pi * f.index.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * f.index.hour / 24)
        f["dow_sin"] = np.sin(2 * np.pi * f.index.dayofweek / 7)
        f["dow_cos"] = np.cos(2 * np.pi * f.index.dayofweek / 7)
        features = ["load", "delta", "delta2", rolling_col,
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        short_period, long_period = 24, 168
    elif freq == "d":
        rolling_window, lag_short, lag_long = 7, 1, 7
        rolling_col, lag_short_col, lag_long_col = "rolling_std_7", "lag1", "lag7"
        f["dow_sin"] = np.sin(2 * np.pi * f.index.dayofweek / 7)
        f["dow_cos"] = np.cos(2 * np.pi * f.index.dayofweek / 7)
        f["month_sin"] = np.sin(2 * np.pi * f.index.month / 12)
        f["month_cos"] = np.cos(2 * np.pi * f.index.month / 12)
        features = ["load", "delta", "delta2", rolling_col,
                    "dow_sin", "dow_cos", "month_sin", "month_cos"]
        short_period, long_period = 7, 1
    else:
        raise ValueError("freq must be 'h' or 'd'")

    f["delta"] = f["load"].diff()
    f["delta2"] = f["delta"].diff()
    f[rolling_col] = f["load"].rolling(rolling_window, min_periods=rolling_window).std()
    f[lag_short_col] = f["load"].shift(lag_short)
    f[lag_long_col] = f["load"].shift(lag_long)

    n_fit = max(1, int(len(f) * fit_frac))
    fit = f.iloc[:n_fit]
    thresholds = {
        "peak_p90": float(fit["load"].quantile(0.90)),
        "valley_p10": float(fit["load"].quantile(0.10)),
        "ramp_p90": float(fit["delta"].abs().quantile(0.90)),
    }
    f["is_peak"] = (f["load"] >= thresholds["peak_p90"]).astype("int64")
    f["is_valley"] = (f["load"] <= thresholds["valley_p10"]).astype("int64")
    f["is_ramp"] = (f["delta"].abs() >= thresholds["ramp_p90"]).astype("int64")
    # Mutually exclusive target classes. Peak takes precedence over Ramp.
    f["state"] = 0
    f.loc[f["is_ramp"].eq(1), "state"] = 2
    f.loc[f["is_peak"].eq(1), "state"] = 1

    x_mean = fit[features].mean()
    x_std = _safe_std(fit[features])
    y_mean = float(fit["load"].mean())
    y_std = float(fit["load"].std())
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    f["target_z"] = (f["load"] - y_mean) / y_std
    f[features] = (f[features] - x_mean) / x_std
    f[f"{lag_short_col}_z"] = (f[lag_short_col] - y_mean) / y_std
    f[f"{lag_long_col}_z"] = (f[lag_long_col] - y_mean) / y_std
    return f, features, y_mean, y_std, thresholds


class WindowDataset(Dataset):
    """Each sample ends at t-1 and predicts target/state at t."""
    def __init__(self, frame: pd.DataFrame, features: Sequence[str],
                 seq_len: int = 168, pred_len: int = 1,
                 p_cols: Optional[Sequence[str]] = None,
                 state_col: str = "state"):
        if seq_len < 1 or pred_len != 1:
            raise ValueError("seq_len must be positive; this implementation supports pred_len=1")
        required = list(features) + ["target_z", state_col]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns required for windowing: {missing}")
        self.seq_len = seq_len
        self.frame = frame
        self.target_indices = []
        self.x, self.y, self.states = [], [], []
        values = frame[list(features)].to_numpy(dtype=np.float32)
        targets = frame["target_z"].to_numpy(dtype=np.float32)
        states = frame[state_col].to_numpy(dtype=np.int64)
        index = frame.index
        for target_i in range(seq_len, len(frame)):
            start = target_i - seq_len
            if not np.isfinite(values[start:target_i]).all() or not np.isfinite(targets[target_i]):
                continue
            if not np.isfinite(values[target_i - 1]).all():
                continue
            if isinstance(index, pd.DatetimeIndex):
                expected = pd.date_range(index[start], periods=seq_len + 1,
                                         freq=pd.infer_freq(index[max(0, start-2):min(len(index), target_i+2)]) or
                                         (index[1] - index[0]))
                if not expected.equals(index[start:target_i + 1]):
                    continue
            self.x.append(values[start:target_i])
            self.y.append([targets[target_i]])
            self.states.append(states[target_i])
            self.target_indices.append(target_i)
        self.x = torch.as_tensor(np.asarray(self.x), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(self.y), dtype=torch.float32)
        self.states = torch.as_tensor(np.asarray(self.states), dtype=torch.long)
        self.target_indices = np.asarray(self.target_indices, dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.states[idx]


# Legacy name retained to reduce import breakage; p/meta are no longer model inputs.
window_dataset = WindowDataset


def split_dataset(ds, ratios=(0.70, 0.15, 0.15)):
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("ratios must be three non-negative values summing to 1")
    n = len(ds)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (Subset(ds, range(0, n_train)),
            Subset(ds, range(n_train, n_train + n_val)),
            Subset(ds, range(n_train + n_val, n)))


def chronological_split(ds, ratios=(0.70, 0.15, 0.15)):
    """Explicit alias documenting that split_dataset preserves chronological order."""
    return split_dataset(ds, ratios)
