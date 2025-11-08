
import pandas as pd
df = pd.read_parquet("processed/NSE_ABB, 1D (1)_merged.parquet")
df.to_csv("processed/NSE_ABB, 1D (1)_merged.csv", index=False)
print(df.columns.tolist())