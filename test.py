from dataset.dataset_windows import build_windows_from_paths
from graph.graph_builder import build_adj_for_window_from_parquet_dfs
from timeseries_models.encoders import CombinedEncoder
import torch, numpy as np, glob, pandas as pd

# get parquet paths
paths = sorted(glob.glob("processed/*_merged.parquet"))
# pick numeric columns (drop Date)
sample_df = pd.read_parquet(paths[0])
feature_cols = [c for c in sample_df.columns if c != "Date"]

# build a small window set
X, dates = build_windows_from_paths(paths, feature_cols, W=180)
X_sample = torch.tensor(X[0], dtype=torch.float32)   # (N,W,F)
# build adjacency for the same window
dfs = [pd.read_parquet(p) for p in paths]
slice_dates = sample_df["Date"].iloc[:180]
A = torch.tensor(build_adj_for_window_from_parquet_dfs(dfs, slice_dates), dtype=torch.float32)
print(A)

# pass through encoder
model = CombinedEncoder(input_dim=X_sample.shape[-1])
embeds = model(X_sample, A)
print(embeds.shape)   # should be (N, d_gnn)
print(embeds)