# feature_pipeline/macro_features_weekly.py
"""
Aggregate daily macro features to weekly summary features used by meta-agent.
Resample weekly (W-FRI): compute last, mean, std for each daily feature.
"""

import pandas as pd
from typing import List

def aggregate_weekly(daily_macro_df: pd.DataFrame, date_col: str = "Date"):
    df = daily_macro_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    weekly = df.resample("W-FRI").agg(['last', 'mean', 'std'])
    # flatten multiindex
    weekly.columns = [f"{c[0]}_{c[1]}" for c in weekly.columns]
    weekly = weekly.reset_index()
    weekly = weekly.fillna(method="ffill").fillna(0.0)
    return weekly

def select_meta_features(weekly_df: pd.DataFrame, include_patterns: List[str] = None):
    if include_patterns is None:
        return weekly_df
    cols = ["Date"]
    for p in include_patterns:
        cols += [c for c in weekly_df.columns if p in c]
    # ensure unique order
    cols = list(dict.fromkeys(cols))
    return weekly_df[cols]

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/macros/combined_macros.csv", parse_dates=["Date"])
    weekly = aggregate_weekly(df)
    print(weekly.head())
