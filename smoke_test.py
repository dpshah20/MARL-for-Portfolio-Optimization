# test.py
from dataset.dataset_windows import build_windows_from_paths
from feature_pipeline.macros_preprocessor import build_macro_features
from feature_pipeline.macros_features_weekly import aggregate_weekly
from timeseries_models.encoders import CombinedEncoder
from graph.graph_builder import build_adj_for_window_from_parquet_dfs
import glob, torch, pandas as pd

def smoke_test():
    paths = sorted(glob.glob("processed/*_merged.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError("No processed parquet files found under processed/")

    sample_df = pd.read_parquet(paths[0])
    feature_cols = [c for c in sample_df.columns if c != "Date"]

    print(f"Building windows for {len(paths[:5])} stocks...")
    X, dates = build_windows_from_paths(paths[:5], feature_cols, W=126)
    X0 = torch.tensor(X[0], dtype=torch.float32)

    dfs = [pd.read_parquet(p) for p in paths[:5]]
    slice_dates = dfs[0]["Date"].iloc[:126].tolist()
    A = torch.tensor(build_adj_for_window_from_parquet_dfs(dfs, slice_dates), dtype=torch.float32)

    print("Building encoder...")
    model = CombinedEncoder(input_dim=X0.shape[-1])
    z = model(X0, A)
    print("✅ Encoder output shape:", z.shape)

if __name__ == "__main__":
    smoke_test()
