
import pandas as pd
df = pd.read_parquet("processed/NSE_ABB, 1D (1)_merged.parquet")
print(df.columns.tolist())