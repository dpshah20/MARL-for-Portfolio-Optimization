import yfinance as yf
import pandas as pd

# Yahoo Finance ticker for NIFTY 100
ticker = "^NSEI"

# Date range
start_date = "2015-01-01"
end_date = "2025-10-01"

# Download data
df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=False
)

# Reset index so Date becomes a column
df = df.reset_index()

# Keep only OHLCV
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Drop any missing rows
df = df.dropna()

# Sort just in case
df = df.sort_values("Date")

# Save final processed file
output_file = "nifty50_ohlcv_2015_2025.csv"
df.to_csv(output_file, index=False)

print("Saved file:", output_file)
print("Rows:", len(df))
print(df.head())