"""Technical feature computation module"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator

def compute_technical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute technical indicators for OHLCV data"""
    df = df.copy()
    
    # Simple Moving Averages
    for period in cfg.MA_PERIODS:
        sma = SMAIndicator(df["Close"], window=period)
        df[f"SMA{period}"] = sma.sma_indicator()
    
    # RSI
    rsi = RSIIndicator(df["Close"], window=cfg.RSI_PERIOD)
    df["RSI14"] = rsi.rsi()
    
    # MACD
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    
    # ADX
    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()
    
    # Volume indicators
    if "Volume" in df.columns:
        obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
        df["OBV"] = obv.on_balance_volume()
        ad = AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"])
        df["A_D"] = ad.acc_dist_index()
    
    # Volatility
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=cfg.ATR_PERIOD)
    df["ATR"] = atr.average_true_range()
    
    bb = BollingerBands(df["Close"], window=cfg.BBANDS_PERIOD)
    df["Boll_Bandwidth"] = bb.bollinger_pband()
    
    # Forward fill NaN values from indicators
    df = df.fillna(method="ffill").fillna(0)
    
    return df


