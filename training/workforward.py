# training/walkforward.py
"""
Lightweight walk-forward schema:
 - split data into rolling windows
 - for each window: train on train_period, validate on val_period
This is a helper wrapper that you can flesh out later.
"""

from typing import List, Tuple
import pandas as pd

def simple_walkforward(dates, train_size: int, val_size: int):
    """
    dates: list of dates (sorted)
    yields (train_dates, val_dates)
    """
    i = 0
    n = len(dates)
    while i + train_size + val_size <= n:
        train = dates[i : i + train_size]
        val = dates[i + train_size : i + train_size + val_size]
        yield train, val
        i += val_size
