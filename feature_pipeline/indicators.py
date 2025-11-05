# feature_pipeline/indicators.py
import numpy as np
import pandas as pd

def sma(df, period, col="Close", name=None):
    name = name or f"SMA{period}"
    df[name] = df[col].rolling(period).mean()
    return df

def rsi(df, period=14, col="Close", name=None):
    name = name or f"RSI{period}"
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df[name] = 100 - (100 / (1 + rs))
    return df

def macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

def adx(df, period=14):
    high = df["High"]; low = df["Low"]; close = df["Close"]
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["TR"] = tr
    plus_dm = high.diff()
    minus_dm = low.diff()
    # positive DM
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr_sum = df["TR"].rolling(period).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / (tr_sum + 1e-8))
    minus_di = 100 * (pd.Series(-minus_dm).rolling(period).sum() / (tr_sum + 1e-8))
    df["DX"] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
    df["ADX"] = df["DX"].rolling(period).mean()
    return df

def obv(df):
    df["OBV"] = (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()
    return df

def accumulation_distribution(df):
    # CLV * Volume cumulative
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-8)
    df["A_D"] = (clv * df["Volume"]).cumsum()
    return df

def atr(df, period=14):
    tr = pd.concat([(df["High"] - df["Low"]).abs(), 
                    (df["High"] - df["Close"].shift()).abs(), 
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df

def bollinger(df, period=20, std_mult=2):
    ma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["Boll_MA"] = ma
    df["Boll_Upper"] = ma + std_mult * std
    df["Boll_Lower"] = ma - std_mult * std
    df["Boll_Bandwidth"] = (df["Boll_Upper"] - df["Boll_Lower"]) / (ma + 1e-8)
    return df
