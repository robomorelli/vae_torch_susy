import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class PandasDF(Dataset):
    def __init__(self, df, columns, cut = True):
        
        if cut:
            range_mbb = [100, 140]
            range_mct2 = [100, 1000]
            df = df[((df['mbb'] >= range_mbb[0]) & (df['mbb'] < range_mbb[1]))]
            df = df[((df['mct2'] >= range_mct2[0]) & (df['mct2'] < range_mct2[1]))]
            df = df[((df['mt'] >= 0) & (df['mt'] < 1000))]
            df = df[((df['met'] >= 0) & (df['mt'] < 1000))]
            df = df[((df['mlb1'] >= 0) & (df['mlb1'] < 1000))]
            df = df[((df['lep1Pt'] >= 0) & (df['lep1Pt'] < 1000))]
            
        self.x = df[columns].values
        self.X = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]


