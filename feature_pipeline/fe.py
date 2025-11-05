# feature_pipeline/feature_engineering.py
import pandas as pd
from .indicators import sma, rsi, macd, adx, obv, accumulation_distribution, atr, bollinger

def compute_technical_features(df, cfg):
    """
    Input: DataFrame with Date, Open, High, Low, Close, Volume
    Returns: DataFrame with indicators (non-normalized)
    """
    df = df.copy()
    # Basic sanity
    df = df.sort_values("Date").reset_index(drop=True)
    df = sma(df, cfg.SMA_SHORT)
    df = sma(df, cfg.SMA_LONG)
    df = rsi(df, cfg.RSI_PERIOD)
    df = macd(df, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
    df = adx(df, cfg.ADX_PERIOD)
    df = obv(df)
    df = accumulation_distribution(df)
    df = atr(df, cfg.ATR_PERIOD)
    df = bollinger(df, cfg.BOLL_PERIOD, cfg.BOLL_STD)

    # fill and drop helper cols
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)
    df.drop(columns=["TR", "DX"], errors="ignore", inplace=True)
    return df

def zscore_normalize(df, numeric_cols, window=252):
    df = df.copy()
    for c in numeric_cols:
        mu = df[c].rolling(window, min_periods=20).mean()
        sd = df[c].rolling(window, min_periods=20).std().replace(0, 1)
        df[c + "_zn"] = (df[c] - mu) / sd
        df[c + "_zn"] = df[c + "_zn"].fillna(0)
    return df
