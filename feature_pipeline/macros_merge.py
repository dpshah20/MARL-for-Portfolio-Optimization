# feature_pipeline/macro_merge.py
import os
import glob
import pandas as pd
from .utils_io import read_csv_flex, save_parquet
from . import config as cfg

def read_macro(path):
    df = read_csv_flex(path)
    # column renamed to basename_Close
    name = os.path.basename(path).replace(".csv","").replace(" ", "_")
    col_name = f"{name}_Close"
    df = df.rename(columns={"Close": col_name})
    df = df[["Date", col_name]]
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").ffill().fillna(0)
    df[name + "_ret"] = df[col_name].pct_change().fillna(0)
    return df[["Date", name + "_ret"]]

def merge_macros_to_stock(stock_parquet_path, macro_dir=cfg.MACROS_DIR, out_dir=cfg.PROCESSED_DIR):
    stock_name = os.path.basename(stock_parquet_path).replace(".parquet","")
    stock_df = pd.read_parquet(stock_parquet_path).sort_values("Date").reset_index(drop=True)
    master_dates = stock_df["Date"].unique()
    macro_files = glob.glob(os.path.join(macro_dir, "*.csv"))
    macro_dfs = []
    for m in macro_files:
        md = read_macro(m)
        md = md.set_index("Date").reindex(master_dates).sort_index().reset_index()
        # forward fill and rename col to <macro>_ret
        col = [c for c in md.columns if c != "Date"][0]
        md[col] = pd.to_numeric(md[col], errors="coerce").ffill().fillna(0)
        macro_dfs.append(md[["Date", col]])
    # merge all macros into one df
    if macro_dfs:
        merged_macro = macro_dfs[0]
        for md in macro_dfs[1:]:
            merged_macro = merged_macro.merge(md, on="Date", how="left")
    else:
        merged_macro = pd.DataFrame({"Date": master_dates})
    merged = stock_df.merge(merged_macro, on="Date", how="left")
    # fill any remaining NaNs for macro cols
    macro_cols = [c for c in merged.columns if c.endswith("_ret")]
    for c in macro_cols:
        merged[c] = merged[c].fillna(0)
    out_path = os.path.join(out_dir, f"{stock_name}_merged.parquet")
    save_parquet(merged, out_path)
    print(f"[macro_merge] saved {out_path}")
    return out_path

def merge_all_stocks_with_macros(processed_dir=cfg.PROCESSED_DIR, macro_dir=cfg.MACROS_DIR, out_dir=cfg.PROCESSED_DIR):
    import glob
    parquet_files = glob.glob(os.path.join(processed_dir, "*.parquet"))
    out_paths = []
    for p in parquet_files:
        # skip already merged (endswith _merged.parquet)
        if p.endswith("_merged.parquet"):
            continue
        outp = merge_macros_to_stock(p, macro_dir=macro_dir, out_dir=out_dir)
        out_paths.append(outp)
    return out_paths

if __name__ == "__main__":
    merge_all_stocks_with_macros()
