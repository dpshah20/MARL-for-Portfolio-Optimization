import pandas as pd

df = pd.read_parquet("processed/NSE_ABB, 1D (1)_merged.parquet")
print(df.columns)
print(df.head(5))
df.to_csv("NSE_ABB_1D_merged.csv", index=False)
print("Total rows:", len(df))