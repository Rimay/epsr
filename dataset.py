import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython import embed

DATA_PATH = "C:/Users/Administrator/Desktop/data/"

# 加载uci数据集，返回原始数据, 日数据, 周数据
def load_uci():
    # 加载原始txt文件
    df = pd.read_csv(DATA_PATH+"uci/household_power_consumption.txt",\
         sep=';', header=0, low_memory=False, engine='c', na_values='?')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    # 数据预处理, 包括缺失值处理,下采样, 季节性分解
    df.dropna(inplace=True)
    dfh, dfd = df['Global_active_power'].resample('h').mean(),\
                 df['Global_active_power'].resample('d').mean()
    dfh.dropna(inplace=True)
    dfd.dropna(inplace=True)
    # dfh.plot()
    # dfd.plot()
    # plt.show()
    # decom = seasonal_decompose(dfd, model='multiplicative', period=12)
    # decom.plot()
    # plt.show()

    return df, dfd, dfh


class uci_dataset(Dataset):
    def __init__(self, data, seq_len=7, pred_len=1):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.x, self.y = [], []
        
        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            self.x.append(data[i : i+seq_len])
            self.y.append(data[i+seq_len : i+seq_len+pred_len])   
        
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_GEFCom2014():
    # df = pd.read_csv(DATA_PATH+'GEFCom2014/load', sep=';', index_col=0)
    pass

# 加载GEFCom2014数据集
class GEFCom2014_dataset(Dataset):
    def __init__(self, data, seq_len=7, pred_len=1): # todo: 更新默认值
        # self.seq_len, self.pred_len = seq_len, pred_len
        # self.x, self.y = [], []
        
        # for i in range(len(data) - self.seq_len - self.pred_len + 1):
        #     self.x.append(data[i : i+seq_len])
        #     self.y.append(data[i+seq_len : i+seq_len+pred_len])   
        
        # self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        # self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        pass

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
