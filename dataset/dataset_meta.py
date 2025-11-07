# dataset/dataset_meta.py
"""
Builds meta features dataset for the Meta-Agent.
Combines weekly macro features (from combined_macros_weekly.csv)
with reinforcement learning reward statistics (from training_logs.csv).
"""

import pandas as pd
import numpy as np
import os

def build_meta_dataset(macros_path: str, rl_logs_path: str):
    """
    Combine macro-level features and RL reward stats into one weekly dataset.

    Args:
        macros_path: path to 'data/macros/combined_macros_weekly.csv'
        rl_logs_path: path to 'logging/training_logs.csv'

    Returns:
        meta_states: np.ndarray of weekly macro feature vectors
        reward_stats: np.ndarray of per-week RL performance metrics (reward, nav change, etc.)
    """
    # --- Load macro features ---
    if not os.path.exists(macros_path):
        raise FileNotFoundError(f"Macro features file not found: {macros_path}")
    df_macros = pd.read_csv(macros_path, parse_dates=["Date"]).sort_values("Date")

    # --- Load RL logs ---
    if not os.path.exists(rl_logs_path):
        raise FileNotFoundError(f"RL logs file not found: {rl_logs_path}")
    df_rl = pd.read_csv(rl_logs_path)
    # Clean and group by week
    if "date" in df_rl.columns:
        df_rl["date"] = pd.to_datetime(df_rl["date"], errors="coerce")
        df_rl = df_rl.dropna(subset=["date"])
        df_rl["week"] = df_rl["date"].dt.to_period("W-FRI").dt.start_time
        weekly_rl = df_rl.groupby("week").agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            nav_last=("nav", "last")
        ).reset_index()
    else:
        raise ValueError("RL logs missing 'date' column")

    # --- Merge macro features with weekly RL metrics ---
    df_combined = pd.merge(df_macros, weekly_rl, left_on="Date", right_on="week", how="inner")
    df_combined = df_combined.drop(columns=["week"], errors="ignore").fillna(0.0)

    # --- Extract features ---
    feature_cols = [c for c in df_combined.columns if c not in ["Date", "reward_mean", "reward_std", "nav_last"]]
    meta_states = df_combined[feature_cols].to_numpy(dtype=np.float32)
    reward_stats = df_combined[["reward_mean", "reward_std", "nav_last"]].to_numpy(dtype=np.float32)

    print(f"[MetaDataset] Loaded {len(df_combined)} weekly samples | Features={len(feature_cols)}")
    return meta_states, reward_stats
