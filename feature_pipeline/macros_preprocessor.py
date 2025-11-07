# feature_pipeline/macros_preprocess.py
"""
Load macro OHLCV CSVs and compute daily macro features:
 - log return (1d)
 - rolling volatility (5d)
 - momentum (21d)
 - normalized VIX level (zscore)
Output: DataFrame with Date and columns like {SYM}_ret, {SYM}_vol5, {SYM}_mom21, {SYM}_close
"""

import pandas as pd
import numpy as np
import os
from typing import Dict

def load_macro_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df.rename(columns={date_col: "Date"}, inplace=True)
    return df

def compute_features_for_symbol(df: pd.DataFrame, price_col: str = "Close",
                                vol_window: int = 5, mom_window: int = 21) -> pd.DataFrame:
    dfc = df[["Date", price_col]].copy()
    dfc["ret"] = np.log(dfc[price_col] / dfc[price_col].shift(1))
    dfc["vol5"] = dfc["ret"].rolling(vol_window, min_periods=1).std().fillna(0.0)
    dfc["mom21"] = dfc[price_col].pct_change(periods=mom_window).fillna(0.0)
    return dfc

def build_macro_features(paths: Dict[str, str],
                         price_col: str = "Close",
                         vol_window: int = 5,
                         mom_window: int = 21) -> pd.DataFrame:
    # paths: dict symbol -> csv path
    out = None
    for sym, p in paths.items():
        df = load_macro_csv(p)
        feats = compute_features_for_symbol(df, price_col=price_col, vol_window=vol_window, mom_window=mom_window)
        feats = feats.rename(columns={
            price_col: f"{sym}_close",
            "ret": f"{sym}_ret",
            "vol5": f"{sym}_vol{vol_window}",
            "mom21": f"{sym}_mom{mom_window}"
        })
        if out is None:
            out = feats
        else:
            out = out.merge(feats, on="Date", how="outer")
    out = out.sort_values("Date").reset_index(drop=True)
    # forward/backfill minimal gaps
    out = out.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return out

if __name__ == "__main__":
    sample = {
        "NIFTY50": "data/macros/NIFTY50.csv",
        "NASDAQ": "data/macros/NASDAQ.csv",
        "SP500": "data/macros/SP500.csv",
        "USDINR": "data/macros/USDINR.csv",
        "INDIAVIX": "data/macros/INDIAVIX.csv",
        "CRUDE": "data/macros/CRUDE.csv"
    }
    df = build_macro_features(sample)
    print(df.head())
