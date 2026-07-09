import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython import embed

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

# todo 添加electric load diagrams 2011-2014数据集
class eld_dataset(Dataset):
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
