import pandas as pd

# 🔹 Path to your parquet file (update as needed)
file_path = "processed/NSE_ABB, 1D (1)_merged.parquet"

# 🔹 Load only the schema, not the entire dataset (for speed)
df = pd.read_parquet(file_path)

# 🔹 Print column names
print("Total Columns:", len(df.columns))
print("Columns:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")
