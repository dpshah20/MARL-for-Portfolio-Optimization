# dataset/dataset_windows.py
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiStockWindowDataset(Dataset):
    """
    Creates windows (W) per stock for each calendar day present in all stocks.
    Inputs: list of merged parquet files (stock_merged.parquet), feature_cols (ordered).
    Returns per item:
      - X: tensor (N, W, F) where N = number of stocks in the list
      - macro: tensor (M,) macro vector for that day
      - date: pd.Timestamp
      - stock_names: list[str]
    """
    def __init__(self, parquet_paths, feature_cols, W=30, min_date=None):
        self.paths = sorted(parquet_paths)
        self.feature_cols = feature_cols
        self.W = W
        # load dataframes (keep minimal columns to save memory)
        self.dfs = {}
        for p in self.paths:
            name = os.path.basename(p).replace("_merged.parquet","").replace(".parquet","")
            df = pd.read_parquet(p)[["Date"] + feature_cols].copy()
            df = df.sort_values("Date").reset_index(drop=True)
            self.dfs[name] = df
        # compute intersection of dates
        date_sets = [set(df["Date"].values) for df in self.dfs.values()]
        self.dates = sorted(list(set.intersection(*date_sets)))
        if min_date:
            self.dates = [d for d in self.dates if d >= pd.to_datetime(min_date)]
        # precompute arrays for speed
        self.names = list(self.dfs.keys())
        self.arrs = {}
        for name in self.names:
            df = self.dfs[name]
            self.arrs[name] = {"dates": df["Date"].values, "arr": df[feature_cols].to_numpy(dtype=np.float32)}
        # valid dates where each stock has at least W history
        self.valid_dates = []
        for d in self.dates:
            ok = True
            for name in self.names:
                idxs = np.where(self.arrs[name]["dates"] == d)[0]
                if len(idxs)==0:
                    ok=False; break
                pos = idxs[0]
                if pos - (self.W-1) < 0:
                    ok=False; break
            if ok:
                self.valid_dates.append(d)

    def __len__(self):
        return len(self.valid_dates)

    def __getitem__(self, idx):
        date = self.valid_dates[idx]
        N = len(self.names)
        F = len(self.feature_cols)
        X = np.zeros((N, self.W, F), dtype=np.float32)
        macro_cols = [c for c in self.feature_cols if c.endswith("_ret")]
        for i, name in enumerate(self.names):
            arr = self.arrs[name]["arr"]
            dates = self.arrs[name]["dates"]
            pos = np.where(dates == date)[0][0]
            X[i] = arr[pos-self.W+1:pos+1, :]
        # macro vector: from first stock (macro cols identical across merged files)
        if macro_cols:
            first_arr = self.arrs[self.names[0]]["arr"]
            # find index of macro cols in feature_cols
            idxs = [self.feature_cols.index(c) for c in macro_cols]
            pos0 = np.where(self.arrs[self.names[0]]["dates"] == date)[0][0]
            macro_vec = first_arr[pos0, idxs].astype(np.float32)
        else:
            macro_vec = np.zeros((0,), dtype=np.float32)
        return {
            "X": torch.from_numpy(X),         # (N, W, F)
            "macro": torch.from_numpy(macro_vec),
            "date": date,
            "stock_names": self.names
        }
