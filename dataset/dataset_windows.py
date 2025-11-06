"""
dataset_windows.py

Functions to build sliding windows from a list of merged parquet files (one per stock).
Outputs a numpy array or generator with shape conventions:
  - Full dataset X shape: (T, N, W, F)
  - Single-day window: X[t] -> (N, W, F)
Where:
  T = number of sliding windows (num_days - W)
  N = number of stocks (len(paths))
  W = window length (days, default 180)
  F = number of numeric features per day
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def load_merged_parquets(paths: List[str]) -> List[pd.DataFrame]:
    """Read parquet files and return list of dataframes sorted by Date."""
    dfs = []
    for p in paths:
        df = pd.read_parquet(p).sort_values("Date").reset_index(drop=True)
        dfs.append(df)
    return dfs

def common_dates(dfs: List[pd.DataFrame]) -> List[pd.Timestamp]:
    """Return sorted intersection of Date values across all dfs."""
    sets = [set(df["Date"].values) for df in dfs]
    common = sorted(list(set.intersection(*sets)))
    return common

def build_feature_matrix_for_date_slice(
    dfs: List[pd.DataFrame],
    slice_dates: List[pd.Timestamp],
    feature_cols: List[str],
) -> np.ndarray:
    """
    Build (N, W, F) matrix for given slice_dates (length W).
    Assumes each df contains Date + feature_cols.
    """
    N = len(dfs)
    W = len(slice_dates)
    F = len(feature_cols)
    arr = np.zeros((N, W, F), dtype=np.float32)
    for i, df in enumerate(dfs):
        sub = df[df["Date"].isin(slice_dates)].sort_values("Date")
        # ensure exact order and length
        assert len(sub) == W, f"stock {i} missing dates in slice"
        arr[i] = sub[feature_cols].to_numpy(dtype=np.float32)
    return arr

def build_windows_from_paths(
    parquet_paths: List[str],
    feature_cols: List[str],
    W: int = 180,
    min_date: Optional[str] = None,
    as_numpy: bool = True
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Build the full sliding-window dataset from a list of merged parquet files.

    Returns:
      - X : numpy array with shape (T, N, W, F) if as_numpy True
      - dates : list of pd.Timestamp of length T where each date is the last day of the window
    """
    dfs = load_merged_parquets(parquet_paths)
    common = common_dates(dfs)
    if min_date:
        common = [d for d in common if d >= pd.to_datetime(min_date)]
    # number of windows
    T = len(common) - W
    if T <= 0:
        raise ValueError(f"Not enough common dates ({len(common)}) for window W={W}")
    windows = []
    window_dates = []
    for i in range(T):
        slice_dates = common[i : i + W]
        last_date = common[i + W]  # day after window = index for prediction, or you can use slice_dates[-1] as last day included
        # Our convention: use window whose last day is slice_dates[-1], and date label = slice_dates[-1]
        # So date_label = slice_dates[-1]
        date_label = slice_dates[-1]
        arr = build_feature_matrix_for_date_slice(dfs, slice_dates, feature_cols)
        windows.append(arr)
        window_dates.append(date_label)
    X = np.stack(windows, axis=0)  # (T, N, W, F)
    return (X, window_dates) if as_numpy else (windows, window_dates)

def windows_generator_from_paths(
    parquet_paths: List[str],
    feature_cols: List[str],
    W: int = 180,
    min_date: Optional[str] = None
):
    """
    Generator that yields (date_label, X_day) where X_day: (N, W, F)
    Useful to stream windows instead of loading full (T,N,W,F) into memory.
    """
    dfs = load_merged_parquets(parquet_paths)
    common = common_dates(dfs)
    if min_date:
        common = [d for d in common if d >= pd.to_datetime(min_date)]
    T = len(common) - W
    if T <= 0:
        raise ValueError("Not enough common dates for the chosen window")
    for i in range(T):
        slice_dates = common[i : i + W]
        date_label = slice_dates[-1]
        arr = build_feature_matrix_for_date_slice(dfs, slice_dates, feature_cols)
        yield date_label, arr  # arr shape (N, W, F)
