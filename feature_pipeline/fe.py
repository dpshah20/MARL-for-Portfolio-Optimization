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

    df = df.dropna().reset_index(drop=True)  # drop initial NaNs from indicators
    # fill and drop helper cols
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)
    df.drop(columns=["TR", "DX"], errors="ignore", inplace=True)
    return df


