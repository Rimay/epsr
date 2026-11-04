import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from IPython import embed

DATA_PATH = "C:/Users/Administrator/Desktop/EPSR/data/"


def _shift_back(arr: np.ndarray, lag: int) -> np.ndarray:
    """把 arr 向后平移 lag 步：arr_out[t] = arr[t - lag]，前 lag 行为 NaN。
    用切片而非 np.roll，避免环形泄露（末尾绕到开头，因果上无意义）。"""
    out = np.full_like(arr, np.nan)
    out[lag:] = arr[:-lag]
    return out


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐行余弦相似度：[T, D] × [T, D] → [T]；b 某行 NaN 则输出 NaN。"""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / (den + eps)


class window_dataset(Dataset):
    def __init__(self, f, features, seq_len=168, pred_len=1, p_cols=None, meta_cols=None):
        self.seq_len, self.pred_len = seq_len, pred_len
        
        # p_cols 随频率变化：小时级 ['lag24_z','lag168_z']，
        #                   天级 ['lag1_z','lag7_z']
        p_cols = ['lag24_z', 'lag168_z'] if p_cols is None else p_cols
        meta_cols = ["r_raw", "discrepancy", "similarity", "volatility_z"] \
                    if meta_cols is None else meta_cols

        # 预先取列索引，避免在循环里重复计算
        f_ci = f.columns.get_indexer(features)
        y_ci = f.columns.get_indexer(['target_z'])
        meta_ci = f.columns.get_indexer(meta_cols)
        p_ci = f.columns.get_indexer(p_cols)
        self.x, self.y, self.meta, self.p = [], [], [], []
        for i in range(len(f) - seq_len - pred_len + 1):
            # NaN 检查必须覆盖到目标行，否则前 191 行 meta 的 NaN 会进样本
            if f.iloc[i : i + seq_len + pred_len].isna().any().any():
                continue
            self.x.append(f.iloc[i : i+seq_len, f_ci])
            self.y.append(f.iloc[i+seq_len : i+seq_len+pred_len, y_ci])
            self.meta.append(f.iloc[i+seq_len-1 : i+seq_len, meta_ci])  # meta 取最后一个输入步（预测起点 t-1）：可靠性描述符都含 load[t]
            self.p.append(f.iloc[i+seq_len : i+seq_len+pred_len, p_ci])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        self.meta = torch.tensor(np.array(self.meta), dtype=torch.float32)
        self.p = torch.tensor(np.array(self.p), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.p[idx], self.meta[idx]


def split_dataset(ds, ratios=(0.7, 0.15, 0.15)):
    """按时间顺序切分窗口数据集，返回 (train, val, test) 三个 Subset。
    window_dataset 的样本按目标时刻严格升序生成（i 递增、跳过 NaN），
    所以「按下标切分」等价于「按时间切分」，三者互不重叠、不存在未来样本混入训练集。
    ratios 之和必须为 1。
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 之和必须为 1，收到 {sum(ratios)}")
    n = len(ds)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (Subset(ds, range(0, n_train)),
            Subset(ds, range(n_train, n_train + n_val)),
            Subset(ds, range(n_train + n_val, n)))


# 统一加载入口：返回 (df, dfd, dfh) = 原始数据 / 日均值 / 小时均值，负荷列名统一为 'load'
def load_df(ds_name='uci'):
    if ds_name == 'uci':
        # 加载原始txt文件
        df = pd.read_csv(DATA_PATH+"uci/household_power_consumption.txt",\
                        sep=';', header=0, low_memory=False, engine='c', na_values='?')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
        df.rename(columns={'Global_active_power': 'load'}, inplace=True)
        # 数据预处理, 包括缺失值处理,下采样, 季节性分解
        df.dropna(inplace=True)
        dfh = df['load'].resample('h').mean().to_frame()
        dfd = df['load'].resample('d').mean().to_frame()
    elif ds_name == 'GEFCom2014':
        df = pd.read_csv(DATA_PATH + 'GEFCom2014/Load/clean/gefcom2014_load_merged.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.drop(columns=['source_task'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=['load'], inplace=True)
        dfh = df['load'].resample('h').mean().to_frame()
        dfd = df['load'].resample('d').mean().to_frame()
    else:
        raise ValueError(f"未知数据集: {ds_name}")

    dfh.dropna(inplace=True)
    dfd.dropna(inplace=True)
    return df, dfd, dfh

# 构造特征列, 并计算 reliability 描述符, 返回 (f, features, y_mean, y_std, thresh)
def make_frame(df, freq='h', fit_frac=0.7):
    """df: DataFrame, 索引为 datetime, 含 'load' 列.
       freq='h' 小时级预测：窗口 24/168, lag24/lag168, hour_sin/hour_cos
       freq='d' 天级预测  ：窗口 7/1,   lag1/lag7,     dow_sin/month_sin
       fit_frac: 标准化统计量 / 标签阈值只在该比例的前段（训练段）上拟合
    """
    f = df.copy()
    idx = f.index

    # ═══════════════════════════════════════════════════
    # 0. 根据频率推导所有时间尺度与特征列名
    # ═══════════════════════════════════════════════════
    if freq == 'h':
        W_SHORT, W_LONG = 24, 168               # 相似度：日周期 / 周周期
        LAG_SHORT, LAG_LONG = 24, 168           # 滞后特征：前一日 / 前一周
        rolling_win = 24
        rolling_col = 'rolling_std_24'
        lag_s_col, lag_l_col = 'lag24', 'lag168'
        sim_period_cols = ['hour_sin', 'hour_cos']
        # 周期特征：小时 + 星期
        hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
        f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        f['dow_sin']  = np.sin(2 * np.pi * dow / 7)
        f['dow_cos']  = np.cos(2 * np.pi * dow / 7)
        features = ['load', 'delta', 'delta2', rolling_col, 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    elif freq == 'd':
        W_SHORT, W_LONG = 7, 1                  # 相似度：上周同日 / 昨日
        LAG_SHORT, LAG_LONG = 1, 7              # 滞后特征：昨日 / 上周同日
        rolling_win = 7
        rolling_col = 'rolling_std_7'
        lag_s_col, lag_l_col = 'lag1', 'lag7'
        sim_period_cols = ['dow_sin', 'dow_cos']
        # 周期特征：星期几 + 月份
        dow, month = idx.dayofweek.to_numpy(), idx.month.to_numpy()
        f['dow_sin']   = np.sin(2 * np.pi * dow / 7)
        f['dow_cos']   = np.cos(2 * np.pi * dow / 7)
        f['month_sin'] = np.sin(2 * np.pi * month / 12)
        f['month_cos'] = np.cos(2 * np.pi * month / 12)
        features = ['load', 'delta', 'delta2', rolling_col, 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    else:
        raise ValueError(f"freq 必须是 'h' 或 'd', 收到 {freq}")

    # ═══════════════════════════════════════════════════
    # 1. 构造原始特征列（全部基于原始 load 值，列名随 freq 变化）
    # ═══════════════════════════════════════════════════
    f['delta']  = f['load'].diff()
    f['delta2'] = f['delta'].diff()
    f[rolling_col] = f['load'].rolling(rolling_win, min_periods=rolling_win).std()
    f[lag_s_col]   = f['load'].shift(LAG_SHORT)
    f[lag_l_col]   = f['load'].shift(LAG_LONG)

    # 统计量只在训练时间段（前 fit_frac 行）上拟合，全序列只做 transform；
    # 否则会用未来数据的均值/方差去标准化历史行（统计量泄露）
    n_fit = max(1, int(len(f) * fit_frac))
    fit = f.iloc[:n_fit]

    # ═══════════════════════════════════════════════════
    # 2. 计算 reliability 描述符（全部只读 f，不写 f）
    # ═══════════════════════════════════════════════════
    sim_cols = ['load', 'delta', rolling_col] + sim_period_cols
    sim_mean = fit.loc[:, sim_cols].mean()
    sim_std  = fit.loc[:, sim_cols].std()
    sim_z = ((f[sim_cols] - sim_mean) / sim_std).to_numpy(dtype=np.float64)

    sim_s = _cosine(sim_z, _shift_back(sim_z, W_SHORT), 1e-8)   # 短周期相似度
    sim_l = _cosine(sim_z, _shift_back(sim_z, W_LONG), 1e-8)    # 长周期相似度
    sim = 0.5 * (sim_s + sim_l)  # 周期相似性 [-1, 1]

    scale = f[rolling_col].abs()
    discrepancy = 0.5 * (f['load'] - f[lag_s_col]).abs() / scale\
                + 0.5 * (f['load'] - f[lag_l_col]).abs() / scale  # 周期差异性 [-1, 1]
    r_raw = np.exp(-np.clip(discrepancy.to_numpy(dtype=float), 0, 50)) * ((sim + 1) / 2)    #周期可靠性分数 [0, 1]

    # ═══════════════════════════════════════════════════
    # 3. 写入状态标签（必须在标准化之前！thresholds 是原始 kW / kW·h 的分位数，
    #               此时 f["load"] / f["delta"] 仍是原始单位，两边才能对齐）
    # ═══════════════════════════════════════════════════
    # 分位数同样只在训练段上取，避免用未来分布定义标签阈值
    thresh = {
        'peak_p90':   float(fit['load'].quantile(0.9)),
        'peak_p95':   float(fit['load'].quantile(0.95)),
        'valley_p10': float(fit['load'].quantile(0.1)),
        'ramp_p90':   float(fit['delta'].abs().quantile(0.9)),
    }

    f['is_peak']            = (f['load'] >= thresh['peak_p90']).astype(int)
    f['is_extreme_peak']    = (f['load'] >= thresh['peak_p95']).astype(int)
    f['is_valley']          = (f['load'] <= thresh['valley_p10']).astype(int)
    f['is_ramp']            = (f['delta'].abs() >= thresh['ramp_p90']).astype(int)
    f['state']              =  'Normal'
    f.loc[f['is_valley'].eq(1), 'state'] = 'Valley'
    f.loc[f['is_ramp'].eq(1), 'state'] = 'Ramp'
    f.loc[f['is_peak'].eq(1), 'state'] = 'Peak'

    # ═══════════════════════════════════════════════════
    # 4. 批量标准化 + 写入所有剩余列
    # ═══════════════════════════════════════════════════
    x_mean, x_std = fit.loc[:, features].mean(), fit.loc[:, features].std()
    y_mean, y_std = float(fit['load'].mean()), float(fit['load'].std())

    # 基于原始单位计算，必须在 f[features] 标准化之前
    vol_mean = fit[rolling_col].mean()
    vol_std  = fit[rolling_col].std()
    f['volatility_z'] = (f[rolling_col] - vol_mean) / vol_std
    f['target_z']     = (f['load'] - y_mean) / y_std

    f[features] = (f[features] - x_mean) / x_std
    # 滞后列名随 freq 变化：小时级 lag24_z/lag168_z，天级 lag1_z/lag7_z
    f[f'{lag_s_col}_z'] = (f[lag_s_col] - y_mean) / y_std
    f[f'{lag_l_col}_z'] = (f[lag_l_col] - y_mean) / y_std

    f['similarity'] = sim
    f['discrepancy'] = discrepancy
    f['r_raw'] = r_raw
    return f, features, y_mean, y_std, thresh

