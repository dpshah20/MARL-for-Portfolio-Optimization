"""
feature_pipeline/macros_preprocessor.py

Cleans macro OHLCV CSVs (Investing.com / TradingView format) and computes:
 - daily log returns
 - 5-day volatility
 - 21-day momentum

Outputs:
  data/macros/combined_macros.csv
"""

import pandas as pd
import numpy as np
import os
from typing import Dict

# ----------------------
# Helper: clean volume field like '151.9M', '245K', or '-'
# ----------------------
def parse_volume(v):
    if isinstance(v, str):
        v = v.strip()
        if v in ["-", "", "nan"]:
            return np.nan
        mult = 1.0
        if v.endswith("M"):
            mult = 1e6
            v = v[:-1]
        elif v.endswith("K"):
            mult = 1e3
            v = v[:-1]
        try:
            return float(v.replace(",", "")) * mult
        except ValueError:
            return np.nan
    return v

# ----------------------
# Load and clean an individual macro CSV
# ----------------------
def load_macro_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # ✅ robust date parsing for mixed formats
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="raise")
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    # ✅ normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ✅ detect price column (can be price, close, last, etc.)
    possible_price_cols = ["price", "close", "last", "adj_close", "value"]
    price_col = None
    for c in possible_price_cols:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise KeyError(f"No price-like column found in {path}. Columns: {df.columns}")

    # ✅ clean numbers
    for col in [price_col, "open", "high", "low"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .replace("-", np.nan)
                .astype(float)
            )

    # ✅ clean volume
    if "vol." in df.columns:
        df["vol."] = df["vol."].apply(parse_volume)

    # ✅ clean change %
    if "change_%" in df.columns:
        df["change_%"] = (
            df["change_%"]
            .astype(str)
            .str.replace("%", "")
            .replace("-", np.nan)
            .astype(float)
            / 100.0
        )

    df = df.sort_values("date").reset_index(drop=True)
    df.rename(columns={"date": "Date", price_col: "Price"}, inplace=True)
    return df

# ----------------------
# Feature computation
# ----------------------
def compute_features_for_symbol(df: pd.DataFrame,
                                vol_window: int = 5,
                                mom_window: int = 21) -> pd.DataFrame:
    dfc = df[["Date", "Price"]].copy()
    dfc["ret"] = np.log(dfc["Price"] / dfc["Price"].shift(1))
    dfc["vol5"] = dfc["ret"].rolling(vol_window, min_periods=1).std().fillna(0.0)
    dfc["mom21"] = dfc["Price"].pct_change(periods=mom_window).fillna(0.0)
    return dfc

# ----------------------
# Combine all macros
# ----------------------
def build_macro_features(paths: Dict[str, str],
                         vol_window: int = 5,
                         mom_window: int = 21) -> pd.DataFrame:
    out = None
    for sym, p in paths.items():
        print(f"Processing {sym} -> {p}")
        df = load_macro_csv(p)
        feats = compute_features_for_symbol(df,
                                            vol_window=vol_window,
                                            mom_window=mom_window)
        feats = feats.rename(columns={
            "Price": f"{sym}_close",
            "ret": f"{sym}_ret",
            "vol5": f"{sym}_vol{vol_window}",
            "mom21": f"{sym}_mom{mom_window}"
        })
        if out is None:
            out = feats
        else:
            out = out.merge(feats, on="Date", how="outer")

    out = out.sort_values("Date").reset_index(drop=True)
    out = out.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return out

# ----------------------
# Main entry
# ----------------------
if __name__ == "__main__":
    sample = {
        "NIFTY50": "data/macros/Nifty 50 Historical Data.csv",
        "NASDAQ": "data/macros/NASDAQ Composite Historical Data.csv",
        "SP500": "data/macros/S&P 500 Historical Data (1).csv",
        "USDINR": "data/macros/USD_INR Historical Data.csv",
        "INDIAVIX": "data/macros/India VIX Historical Data.csv",
        "CRUDE": "data/macros/Crude Oil WTI Futures Historical Data.csv"
    }

    df = build_macro_features(sample)
    os.makedirs("data/macros", exist_ok=True)
    output_path = "data/macros/combined_macros.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Combined macro features saved to {output_path}")
    print(df.head())
