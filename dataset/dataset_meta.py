# dataset/dataset_meta.py
"""
Build meta dataset for meta-agent training:
 - merges weekly macro features with internal portfolio weekly stats
 - produces X (features) and y (next-week target return or NAV growth)
"""

import pandas as pd
import numpy as np
from typing import Optional

def merge_macro_internal(weekly_macro: pd.DataFrame, internal_weekly: Optional[pd.DataFrame] = None):
    if internal_weekly is None:
        # create placeholder internal stats zeros
        internal_weekly = pd.DataFrame({
            "Date": weekly_macro["Date"],
            "portfolio_weekly_return": np.zeros(len(weekly_macro)),
            "portfolio_weekly_vol": np.zeros(len(weekly_macro)),
            "portfolio_weekly_mdd": np.zeros(len(weekly_macro))
        })
    merged = weekly_macro.merge(internal_weekly, on="Date", how="left")
    merged = merged.fillna(0.0)
    return merged

def build_meta_dataset(merged_weekly: pd.DataFrame, lookahead_weeks: int = 1):
    df = merged_weekly.copy().sort_values("Date").reset_index(drop=True)
    # compute target = cumulative return over next lookahead_weeks
    if "portfolio_weekly_return" in df.columns:
        ret = df["portfolio_weekly_return"].values
        targets = []
        for i in range(len(df)):
            j = i + 1
            k = min(len(df), i + 1 + lookahead_weeks)
            if j >= k:
                targets.append(0.0)
            else:
                cum = np.prod(1 + ret[j:k]) - 1.0
                targets.append(cum)
        df["meta_target"] = targets
    else:
        df["meta_target"] = 0.0
    return df
