# graph/graph_builder.py
import numpy as np
import pandas as pd

def build_corr_adj(stock_dfs, date_index, lookback=60, method="knn", thr=0.6, k=8):
    """
    stock_dfs: dict{name: df} with columns Date and 'return' or a return column with that name.
    date_index: pd.Timestamp for which adjacency is computed (end date).
    lookback: number of days for correlation
    method: 'knn' or 'threshold'
    thr: threshold for absolute correlation (if threshold method)
    k: number neighbors for knn
    Returns: A_norm (N,N), names (list)
    """
    names = list(stock_dfs.keys())
    mat = []
    for name in names:
        df = stock_dfs[name].sort_values("Date").reset_index(drop=True)
        # pick return series; prefer 'return' then 'ret' columns heuristically
        if "return" in df.columns:
            series = df["return"].values
        else:
            # look for any column ending with '_ret' or 'ret'
            ret_cols = [c for c in df.columns if c.endswith("_ret") or c=="ret"]
            if ret_cols:
                series = df[ret_cols[0]].values
            else:
                raise ValueError(f"No return column in stock df {name}")
        # find position of date_index
        idxs = np.where(df["Date"].values <= date_index)[0]
        if len(idxs) == 0:
            seq = np.zeros(lookback)
        else:
            pos = idxs[-1]
            start = max(0, pos - lookback + 1)
            seq = series[start:pos+1]
            if len(seq) < lookback:
                seq = np.pad(seq, (lookback - len(seq), 0), 'constant')
        mat.append(seq)
    R = np.vstack(mat)  # (N, lookback)
    corr = np.corrcoef(R)
    N = corr.shape[0]
    if method == "threshold":
        A = (np.abs(corr) >= thr).astype(float)
    else:
        A = np.zeros_like(corr)
        for i in range(N):
            # exclude self in knn neighbor selection
            order = np.argsort(-np.abs(corr[i]))
            neighbors = [j for j in order if j != i][:k]
            A[i, neighbors] = 1.0
            A[i, i] = 1.0
    np.fill_diagonal(A, 1.0)
    row_sum = A.sum(axis=1, keepdims=True)
    A_norm = A / np.maximum(row_sum, 1.0)
    return A_norm, names
