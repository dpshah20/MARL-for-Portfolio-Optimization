import yfinance as yf
import pandas as pd

# 1️⃣ Define the ticker for Nifty 50
ticker = "^NSEI"  # Yahoo Finance ticker for Nifty 50

# 2️⃣ Define the time period
start_date = "2015-01-01"
end_date   = "2020-12-31"

# 3️⃣ Download data
nifty = yf.download(ticker, start=start_date, end=end_date)

# 4️⃣ Save to CSV
output_file = "nifty_2015_2020.csv"
nifty.to_csv(output_file)

print(f"✅ Download complete! Data saved to {output_file}")
print(nifty.head())
