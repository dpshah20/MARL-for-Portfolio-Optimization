# feature_pipeline/run_pipeline.py
import os
import sys

if __package__ in (None, ""):
    # Allow running this file directly: `python feature_pipeline/run_pipeline.py`
    # by adding repository root to sys.path and importing as a package.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from feature_pipeline.batch_processor import process_all_stocks
    from feature_pipeline.macros_merge import merge_all_stocks_with_macros
else:
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
