# dataset/dataset_windows.py
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def load_merged_parquets(paths: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_parquet(p).sort_values("Date").reset_index(drop=True)
        dfs.append(df)
    return dfs

def common_dates(dfs: List[pd.DataFrame]) -> List[pd.Timestamp]:
    sets = [set(df["Date"].values) for df in dfs]
    common = sorted(list(set.intersection(*sets)))
    return common

def build_feature_matrix_for_date_slice(
    dfs: List[pd.DataFrame],
    slice_dates: List[pd.Timestamp],
    feature_cols: List[str],
) -> np.ndarray:
    N = len(dfs)
    W = len(slice_dates)
    F = len(feature_cols)
    arr = np.zeros((N, W, F), dtype=np.float32)
    for i, df in enumerate(dfs):
        sub = df[df["Date"].isin(slice_dates)].sort_values("Date")
        if len(sub) != W:
            raise ValueError(f"Stock index {i} missing {W - len(sub)} dates in slice")
        # safe column handling: if column missing, fill zeros
        avail = [c for c in feature_cols if c in sub.columns]
        missing = [c for c in feature_cols if c not in sub.columns]
        if missing:
            # warn once
            print(f"[WARN] stock {i} missing columns: {missing} — filling zeros")
        if avail:
            arr_part = sub[avail].to_numpy(dtype=np.float32)
            # put selected columns into arr at correct indices
            for j, col in enumerate(feature_cols):
                if col in avail:
                    k = avail.index(col)
                    arr[i, :, j] = arr_part[:, k]
                else:
                    arr[i, :, j] = 0.0
        else:
            arr[i, :, :] = 0.0
    return arr

def build_windows_from_paths(
    parquet_paths: List[str],
    feature_cols: List[str],
    W: int = 126,
    min_date: Optional[str] = None,
    as_numpy: bool = True
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    dfs = load_merged_parquets(parquet_paths)
    common = common_dates(dfs)
    if min_date:
        common = [d for d in common if d >= pd.to_datetime(min_date)]
    T = len(common) - W
    if T <= 0:
        raise ValueError("Not enough common dates")
    windows = []
    window_dates = []
    for i in range(T):
        slice_dates = common[i:i+W]
        date_label = slice_dates[-1]
        arr = build_feature_matrix_for_date_slice(dfs, slice_dates, feature_cols)
        windows.append(arr)
        window_dates.append(date_label)
    X = np.stack(windows, axis=0)
    return (X, window_dates) if as_numpy else (windows, window_dates)

def windows_generator_from_paths(
    parquet_paths: List[str],
    feature_cols: List[str],
    W: int = 126,
    min_date: Optional[str] = None
):
    dfs = load_merged_parquets(parquet_paths)
    common = common_dates(dfs)
    if min_date:
        common = [d for d in common if d >= pd.to_datetime(min_date)]
    T = len(common) - W
    if T <= 0:
        raise ValueError("Not enough common dates")
    for i in range(T):
        slice_dates = common[i:i+W]
        date_label = slice_dates[-1]
        arr = build_feature_matrix_for_date_slice(dfs, slice_dates, feature_cols)
        yield date_label, arr
