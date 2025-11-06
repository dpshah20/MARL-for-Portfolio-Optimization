"""
graph_builder.py

Functions to build adjacency matrices from returns or embeddings for each window/date.
Returns adjacency in numpy (N,N). Designed to be called on a per-window basis.

Options:
 - method='corr_threshold' : compute Pearson correlation of returns in the window and threshold
 - method='knn' : connect top-k correlated neighbors per node
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def returns_matrix_from_dfs(dfs: List[pd.DataFrame], slice_dates: List[pd.Timestamp]) -> np.ndarray:
    """
    Given list of dfs (each has Date, Close), compute returns series per stock for slice_dates.
    Returns shape (N, W-1) as pct_change over window days.
    """
    arrs = []
    for df in dfs:
        sub = df[df["Date"].isin(slice_dates)].sort_values("Date")
        prices = sub["Close"].to_numpy(dtype=np.float64)
        rets = np.diff(prices) / prices[:-1]
        arrs.append(rets)
    return np.stack(arrs, axis=0)  # (N, W-1)

def correlation_adj_from_returns(
    returns_mat: np.ndarray,
    method: str = "knn",
    thr: float = 0.6,
    k: int = 8,
    absolute: bool = True
) -> np.ndarray:
    """
    Build adjacency from returns matrix.
    returns_mat: (N, T) where T = window_length-1
    method: 'corr_threshold' or 'knn'
    thr: threshold for abs(corr)
    k: neighbors for knn (per node)
    absolute: take absolute correlation if True
    Returns: A (N,N) row-normalized adjacency (rows sum to 1).
    """
    N = returns_mat.shape[0]
    corr = np.corrcoef(returns_mat)  # (N,N)
    if absolute:
        corr = np.abs(corr)
    # fill nan with 0
    corr = np.nan_to_num(corr, nan=0.0)
    A = np.zeros_like(corr, dtype=float)
    if method == "corr_threshold":
        A[corr >= thr] = 1.0
    elif method == "knn":
        for i in range(N):
            order = np.argsort(-corr[i])  # descending
            # exclude self in selection if desired
            neighbors = [j for j in order if j != i][:k]
            A[i, neighbors] = 1.0
    else:
        raise ValueError("Unknown method")
    # ensure at least self-loop (optional)
    np.fill_diagonal(A, 1.0)
    # row-normalize
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    A_norm = A / row_sum
    return A_norm

def build_adj_for_window_from_parquet_dfs(
    dfs: List[pd.DataFrame],
    slice_dates: List[pd.Timestamp],
    method: str = "knn",
    thr: float = 0.6,
    k: int = 8,
    absolute: bool = True
) -> np.ndarray:
    """
    Convenience: compute returns for the slice and return adjacency matrix A (N,N).
    """
    returns_mat = returns_matrix_from_dfs(dfs, slice_dates)  # (N, W-1)
    A = correlation_adj_from_returns(returns_mat, method=method, thr=thr, k=k, absolute=absolute)
    return A
