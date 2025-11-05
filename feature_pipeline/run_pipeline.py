# feature_pipeline/run_pipeline.py
from .batch_processor import process_all_stocks
from .macros_merge import merge_all_stocks_with_macros

def run_full_pipeline():
    print("==> Processing raw stock CSVs (technical features)...")
    processed = process_all_stocks()
    print("==> Merging macro features into each processed stock file...")
    merged = merge_all_stocks_with_macros()
    print("==> DONE. Outputs in processed/")

if __name__ == "__main__":
    run_full_pipeline()
