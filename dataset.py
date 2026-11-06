import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

DATA_PATH = "C:/Users/Administrator/Desktop/EPSR/data/"


def _shift_back(arr: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    out[lag:] = arr[:-lag]
    return out


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.nansum(a * b, axis=1)
    den = np.linalg.norm(np.nan_to_num(a), axis=1) * np.linalg.norm(np.nan_to_num(b), axis=1)
    result = num / (den + eps)
    result[~np.isfinite(result)] = np.nan
    return result


class window_dataset(Dataset):
    def __init__(self, f, features, seq_len=168, pred_len=1, p_cols=None, meta_cols=None):
        self.seq_len, self.pred_len = seq_len, pred_len
        p_cols = ['lag24_z', 'lag168_z'] if p_cols is None else p_cols
        meta_cols = ["r_raw", "discrepancy", "similarity", "volatility_z"] if meta_cols is None else meta_cols
        f_ci = f.columns.get_indexer(features)
        y_ci = f.columns.get_indexer(['target_z'])
        meta_ci = f.columns.get_indexer(meta_cols)
        p_ci = f.columns.get_indexer(p_cols)
        self.x, self.y, self.meta, self.p = [], [], [], []
        # 对每个候选窗口检查自然时间间隔，防止缺失日期被当作相邻时间步。
        for i in range(len(f) - seq_len - pred_len + 1):
            part = f.iloc[i:i + seq_len + pred_len]
            if part.isna().any().any():
                continue
            if isinstance(f.index, pd.DatetimeIndex):
                expected = pd.date_range(f.index[i], periods=seq_len + pred_len, freq=pd.infer_freq(f.index[max(0, i-2):min(len(f.index), i+seq_len+pred_len+2)]) or None)
                if len(expected) != len(part) or not expected.equals(part.index):
                    continue
            self.x.append(f.iloc[i:i + seq_len, f_ci].to_numpy())
            self.y.append(f.iloc[i + seq_len:i + seq_len + pred_len, y_ci].to_numpy())
            self.meta.append(f.iloc[i + seq_len - 1:i + seq_len, meta_ci].to_numpy())
            self.p.append(f.iloc[i + seq_len:i + seq_len + pred_len, p_ci].to_numpy())
        self.x = torch.tensor(np.asarray(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.y), dtype=torch.float32)
        self.meta = torch.tensor(np.asarray(self.meta), dtype=torch.float32)
        self.p = torch.tensor(np.asarray(self.p), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.p[idx], self.meta[idx]


def split_dataset(ds, ratios=(0.7, 0.15, 0.15)):
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 之和必须为 1，收到 {sum(ratios)}")
    n = len(ds)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (Subset(ds, range(0, n_train)),
            Subset(ds, range(n_train, n_train + n_val)),
            Subset(ds, range(n_train + n_val, n)))


def load_df(ds_name='uci'):
    if ds_name == 'uci':
        df = pd.read_csv(DATA_PATH + "uci/household_power_consumption.txt",
                         sep=';', header=0, low_memory=False, engine='c', na_values='?')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
        df.rename(columns={'Global_active_power': 'load'}, inplace=True)
        # 只按目标列筛除坏记录，不因其他测量字段缺失而丢弃有效负荷。
        df = df.dropna(subset=['load']).copy()
        dfh = df['load'].resample('h').mean().to_frame()
        dfd = df['load'].resample('D').mean().to_frame()
    elif ds_name == 'GEFCom2014':
        df = pd.read_csv(DATA_PATH + 'GEFCom2014/Load/clean/gefcom2014_load_merged.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.drop(columns=['source_task'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df = df.dropna(subset=['load']).copy()
        dfh = df['load'].resample('h').mean().to_frame()
        dfd = df['load'].resample('D').mean().to_frame()
    else:
        raise ValueError(f"未知数据集: {ds_name}")

    # 补齐完整日历索引但不对目标值插值；缺失时段对应的窗口将被丢弃。
    if len(dfh):
        dfh = dfh.asfreq('h')
    if len(dfd):
        dfd = dfd.asfreq('D')
    return df, dfd, dfh


def make_frame(df, freq='h', fit_frac=0.7):
    f = df.copy()
    idx = f.index
    if freq == 'h':
        W_SHORT, W_LONG = 24, 168
        LAG_SHORT, LAG_LONG = 24, 168
        rolling_win, rolling_col = 24, 'rolling_std_24'
        lag_s_col, lag_l_col = 'lag24', 'lag168'
        sim_period_cols = ['hour_sin', 'hour_cos']
        hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
        f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        f['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        features = ['load', 'delta', 'delta2', rolling_col, 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    elif freq == 'd':
        W_SHORT, W_LONG = 7, 1
        LAG_SHORT, LAG_LONG = 1, 7
        rolling_win, rolling_col = 7, 'rolling_std_7'
        lag_s_col, lag_l_col = 'lag1', 'lag7'
        sim_period_cols = ['dow_sin', 'dow_cos']
        dow, month = idx.dayofweek.to_numpy(), idx.month.to_numpy()
        f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        f['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        f['month_sin'] = np.sin(2 * np.pi * month / 12)
        f['month_cos'] = np.cos(2 * np.pi * month / 12)
        features = ['load', 'delta', 'delta2', rolling_col, 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    else:
        raise ValueError(f"freq 必须是 'h' 或 'd', 收到 {freq}")

    f['delta'] = f['load'].diff()
    f['delta2'] = f['delta'].diff()
    f[rolling_col] = f['load'].rolling(rolling_win, min_periods=rolling_win).std()
    f[lag_s_col] = f['load'].shift(LAG_SHORT)
    f[lag_l_col] = f['load'].shift(LAG_LONG)

    n_fit = max(1, int(len(f) * fit_frac))
    fit = f.iloc[:n_fit]
    sim_cols = ['load', 'delta', rolling_col] + sim_period_cols
    sim_mean, sim_std = fit[sim_cols].mean(), fit[sim_cols].std().replace(0, 1)
    sim_z = ((f[sim_cols] - sim_mean) / sim_std).to_numpy(dtype=np.float64)
    sim_s = _cosine(sim_z, _shift_back(sim_z, W_SHORT))
    sim_l = _cosine(sim_z, _shift_back(sim_z, W_LONG))
    sim = 0.5 * (sim_s + sim_l)

    scale = f[rolling_col].abs().clip(lower=1e-6)
    discrepancy = 0.5 * (f['load'] - f[lag_s_col]).abs() / scale + 0.5 * (f['load'] - f[lag_l_col]).abs() / scale
    r_raw = np.exp(-np.clip(discrepancy.to_numpy(dtype=float), 0, 50)) * ((sim + 1) / 2)

    thresh = {
        'peak_p90': float(fit['load'].quantile(0.9)),
        'peak_p95': float(fit['load'].quantile(0.95)),
        'valley_p10': float(fit['load'].quantile(0.1)),
        'ramp_p90': float(fit['delta'].abs().quantile(0.9)),
    }
    f['is_peak'] = (f['load'] >= thresh['peak_p90']).astype(int)
    f['is_extreme_peak'] = (f['load'] >= thresh['peak_p95']).astype(int)
    f['is_valley'] = (f['load'] <= thresh['valley_p10']).astype(int)
    f['is_ramp'] = (f['delta'].abs() >= thresh['ramp_p90']).astype(int)
    f['state'] = 'Normal'
    f.loc[f['is_valley'].eq(1), 'state'] = 'Valley'
    f.loc[f['is_ramp'].eq(1), 'state'] = 'Ramp'
    f.loc[f['is_peak'].eq(1), 'state'] = 'Peak'

    x_mean, x_std = fit[features].mean(), fit[features].std().replace(0, 1)
    y_mean, y_std = float(fit['load'].mean()), float(fit['load'].std())
    vol_mean, vol_std = fit[rolling_col].mean(), fit[rolling_col].std()
    f['volatility_z'] = (f[rolling_col] - vol_mean) / (vol_std if vol_std > 0 else 1)
    f['target_z'] = (f['load'] - y_mean) / (y_std if y_std > 0 else 1)
    f[features] = (f[features] - x_mean) / x_std
    f[f'{lag_s_col}_z'] = (f[lag_s_col] - y_mean) / (y_std if y_std > 0 else 1)
    f[f'{lag_l_col}_z'] = (f[lag_l_col] - y_mean) / (y_std if y_std > 0 else 1)
    f['similarity'] = sim
    f['discrepancy'] = np.log1p(discrepancy.clip(lower=0))
    f['r_raw'] = r_raw
    meta_cols = ['r_raw', 'discrepancy', 'similarity', 'volatility_z']
    meta_mean = f.iloc[:n_fit][meta_cols].mean()
    meta_std = f.iloc[:n_fit][meta_cols].std().replace(0, 1)
    f[meta_cols] = (f[meta_cols] - meta_mean) / meta_std
    return f, features, y_mean, y_std, thresh
