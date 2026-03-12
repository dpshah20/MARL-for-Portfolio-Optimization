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

def all_dates(dfs: List[pd.DataFrame]) -> List[pd.Timestamp]:
    sets = [set(df["Date"].values) for df in dfs]
    union = sorted(list(set.union(*sets)))
    return union

def _index_by_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.sort_values("Date").set_index("Date")
    return out

def build_feature_matrix_for_date_slice(
    dfs: List[pd.DataFrame],
    slice_dates: List[pd.Timestamp],
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(dfs)
    W = len(slice_dates)
    F = len(feature_cols)
    arr = np.zeros((N, W, F), dtype=np.float32)
    valid_mask = np.zeros((N,), dtype=bool)

    slice_idx = pd.DatetimeIndex(slice_dates)

    for i, df in enumerate(dfs):
        df_idx = _index_by_date(df)
        has_full_window = slice_idx.isin(df_idx.index).all()
        if not has_full_window:
            continue

        sub = df_idx.reindex(slice_idx)

        # Safe column handling: if column missing, fill zeros.
        avail = [c for c in feature_cols if c in sub.columns]
        missing = [c for c in feature_cols if c not in sub.columns]
        if missing:
            print(f"[WARN] stock {i} missing columns: {missing} - filling zeros")

        if avail:
            arr_part = sub[avail].to_numpy(dtype=np.float32)
            arr_part = np.nan_to_num(arr_part, nan=0.0, posinf=0.0, neginf=0.0)

            # Put selected columns into arr at correct indices.
            for j, col in enumerate(feature_cols):
                if col in avail:
                    k = avail.index(col)
                    arr[i, :, j] = arr_part[:, k]
                else:
                    arr[i, :, j] = 0.0
        else:
            arr[i, :, :] = 0.0

        valid_mask[i] = True

    return arr, valid_mask

def build_windows_from_paths(
    parquet_paths: List[str],
    feature_cols: List[str],
    W: int = 126,
    min_date: Optional[str] = None,
    as_numpy: bool = True,
    return_valid_mask: bool = False,
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    dfs = load_merged_parquets(parquet_paths)
    common = all_dates(dfs)
    if min_date:
        common = [d for d in common if d >= pd.to_datetime(min_date)]
    T = len(common) - W
    if T <= 0:
        raise ValueError("Not enough dates")

    windows = []
    masks = []
    window_dates = []
    for i in range(T):
        slice_dates = common[i:i+W]
        date_label = slice_dates[-1]
        arr, valid_mask = build_feature_matrix_for_date_slice(dfs, slice_dates, feature_cols)
        windows.append(arr)
        masks.append(valid_mask)
        window_dates.append(date_label)

    X = np.stack(windows, axis=0)
    M = np.stack(masks, axis=0)
    if as_numpy:
        return (X, window_dates, M) if return_valid_mask else (X, window_dates)
    return (windows, window_dates, masks) if return_valid_mask else (windows, window_dates)

def windows_generator_from_paths(
    parquet_paths: List[str],
    feature_cols: List[str],
    W: int = 126,
    min_date: Optional[str] = None,
    return_valid_mask: bool = False,
):
    dfs = load_merged_parquets(parquet_paths)
    common = all_dates(dfs)
    if min_date:
        common = [d for d in common if d >= pd.to_datetime(min_date)]
    T = len(common) - W
    if T <= 0:
        raise ValueError("Not enough dates")

    for i in range(T):
        slice_dates = common[i:i+W]
        date_label = slice_dates[-1]
        arr, valid_mask = build_feature_matrix_for_date_slice(dfs, slice_dates, feature_cols)
        if return_valid_mask:
            yield date_label, arr, valid_mask
        else:
            yield date_label, arr
