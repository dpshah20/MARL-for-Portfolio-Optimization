# feature_pipeline/batch_processor.py
import os
import glob
from .utils_io import read_csv_flex, save_parquet
from .fe import compute_technical_features, zscore_normalize
from . import config as cfg

NUMERIC_FEATURES = [
    "SMA20", "SMA50", "RSI14", "MACD", "MACD_signal", "MACD_hist",
    "ADX", "OBV", "A_D", "ATR", "Boll_Bandwidth"
]

def process_single_stock(in_path, out_dir, cfg=cfg):
    stock_name = os.path.basename(in_path).replace(".csv","")
    print(f"[process] {stock_name}")
    df = read_csv_flex(in_path)
    # filter start date
    df = df[df["Date"] >= cfg.START_DATE].copy()
    if df.empty:
        print(f"  -> No data >= {cfg.START_DATE} for {stock_name}, skipping.")
        return None
    df_feat = compute_technical_features(df, cfg)
    df_feat = zscore_normalize(df_feat, NUMERIC_FEATURES, window=252)
    out_path = os.path.join(out_dir, f"{stock_name}.parquet")
    save_parquet(df_feat, out_path)
    print(f"  -> saved: {out_path} rows={len(df_feat)}")
    return out_path

def process_all_stocks(stocks_dir=cfg.STOCKS_DIR, out_dir=cfg.PROCESSED_DIR, cfg=cfg):
    os.makedirs(out_dir, exist_ok=True)
    print("Looking for CSVs in:", stocks_dir)
    files = glob.glob(os.path.join(stocks_dir, "*.csv"))
    print("Found files:", files[:3])  
    out_paths = []
    for f in files:
        p = process_single_stock(f, out_dir, cfg)
        if p:
            out_paths.append(p)
    return out_paths

if __name__ == "__main__":
    process_all_stocks()
