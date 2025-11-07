# feature_pipeline/macro_features_weekly.py
"""
Aggregate daily macro features to weekly summary features used by the meta-agent.

- Input: data/macros/combined_macros.csv (daily)
- Output: data/macros/combined_macros_weekly.csv (weekly)
- Aggregation: Resample weekly (W-FRI)
               Compute last, mean, std for each feature.
- Purpose: Provides macro-level context features for the meta-agent (weekly updates).
"""

import os
import pandas as pd
from typing import List

# ----------------------------------------------------------
# Aggregate daily macro data to weekly resolution
# ----------------------------------------------------------
def aggregate_weekly(daily_macro_df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = daily_macro_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Weekly aggregation (Friday close of each week)
    weekly = df.resample("W-FRI").agg(['last', 'mean', 'std'])

    # Flatten hierarchical columns
    weekly.columns = [f"{c[0]}_{c[1]}" for c in weekly.columns]
    weekly = weekly.reset_index()

    # Fill missing values (e.g. early start weeks)
    weekly = weekly.fillna(method="ffill").fillna(0.0)
    return weekly

# ----------------------------------------------------------
# Select subset of columns relevant for meta-agent
# ----------------------------------------------------------
def select_meta_features(weekly_df: pd.DataFrame, include_patterns: List[str] = None) -> pd.DataFrame:
    if include_patterns is None:
        return weekly_df

    cols = ["Date"]
    for p in include_patterns:
        cols += [c for c in weekly_df.columns if p in c]
    # ensure unique ordering
    cols = list(dict.fromkeys(cols))
    return weekly_df[cols]

# ----------------------------------------------------------
# Main: build + save weekly macro dataset
# ----------------------------------------------------------
if __name__ == "__main__":
    input_path = "data/macros/combined_macros.csv"
    output_path = "data/macros/combined_macros_weekly.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run macros_preprocessor.py first.")

    print(f"📘 Loading daily macro data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["Date"])

    print("📊 Aggregating to weekly frequency...")
    weekly = aggregate_weekly(df)

    # Example: focus on core macro indicators for meta-agent
    include = ["NIFTY50", "NASDAQ", "SP500", "USDINR", "INDIAVIX", "CRUDE"]
    weekly_meta = select_meta_features(weekly, include)

    # Ensure output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    weekly_meta.to_csv(output_path, index=False)
    print(f"✅ Weekly macro features saved to: {output_path}")
    print(weekly_meta.head(5))
