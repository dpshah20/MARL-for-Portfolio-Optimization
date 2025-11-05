# feature_pipeline/utils_io.py
import os
import pandas as pd
import numpy as np

def read_csv_flex(path, date_first=True):
    """
    Robust reader for CSVs that finds date/price/volume columns flexibly.
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume (if present).
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    # find date column
    date_col = next((c for c in cols if c.lower() in ("date", "time", "timestamp")), cols[0])
    df[date_col] = pd.to_datetime(df[date_col].astype(str), dayfirst=date_first, errors="coerce")
    # find numeric cols
    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None
    close_col = find_col([c for c in cols if c.lower() in ("close","price","last")])
    open_col = find_col([c for c in cols if c.lower()=="open"])
    high_col = find_col([c for c in cols if c.lower()=="high"])
    low_col = find_col([c for c in cols if c.lower()=="low"])
    vol_col = find_col([c for c in cols if c.lower() in ("volume","vol.","vol")])

    out = pd.DataFrame({"Date": df[date_col]})
    if close_col is not None:
        out["Close"] = df[close_col].astype(str).str.replace(",","").replace("", np.nan).astype(float)
    else:
        out["Close"] = np.nan
    if open_col is not None: out["Open"] = pd.to_numeric(df[open_col].astype(str).str.replace(",",""), errors="coerce")
    if high_col is not None: out["High"] = pd.to_numeric(df[high_col].astype(str).str.replace(",",""), errors="coerce")
    if low_col is not None: out["Low"] = pd.to_numeric(df[low_col].astype(str).str.replace(",",""), errors="coerce")
    if vol_col is not None:
        vol = df[vol_col].astype(str).fillna("0").str.replace(",","")
        # parse K/M suffix
        def parse_vol(s):
            s = s.strip()
            try:
                if s.endswith("K") or s.endswith("k"): return float(s[:-1]) * 1e3
                if s.endswith("M") or s.endswith("m"): return float(s[:-1]) * 1e6
                return float(s)
            except:
                return 0.0
        out["Volume"] = vol.apply(parse_vol)
    out = out.sort_values("Date").reset_index(drop=True)
    return out

def save_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
