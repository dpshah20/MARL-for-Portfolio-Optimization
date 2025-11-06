import yfinance as yf
import pandas as pd

# Define ticker and date range
ticker = "INR=X"  # USD/INR
start_date = "2015-01-01"
end_date = "2025-11-06"  # None = till today's date

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
data.to_csv("USD_INR_2015_to_today.csv")

print("✅ Data saved as 'USD_INR_2015_to_today.csv'")
print(data.tail())
