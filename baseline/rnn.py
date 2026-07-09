from IPython import embed
import torch
from torch import nn
from torch.nn import functional as F

class GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2,
                    pred_len=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),  
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, in_ch) -> (batch_size, seq_len, in_ch)
        x = x[:, :, None]
        out, h = self.gru(x)
        feature = out[:, -1, :]
        pred = self.fc(feature)
        return pred

class LSTM(nn.Module):
    """
    input: (B, L, C)
    output: (B, pred_len)
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, pred_len=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),  
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )
    def forward(self, x):
        # x: (batch_size, seq_len, in_ch) -> (batch_size, seq_len, in_ch)
        x = x[:, :, None]
        out, (h, c) = self.lstm(x)
        feature = out[:, -1, :]
        pred = self.fc(feature)
        return pred
