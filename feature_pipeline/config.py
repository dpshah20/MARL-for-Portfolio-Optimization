"""Feature pipeline configuration"""

import os
from datetime import datetime

# Directories
STOCKS_DIR = os.path.join("data", "stocks")
PROCESSED_DIR = "processed"
MACROS_DIR = os.path.join("data", "macros")

# Date range
START_DATE = datetime(2015, 1, 1)

# Feature engineering params
PRICE_COLS = ["Open", "High", "Low", "Close"]
VOL_COLS = ["Volume"]
MA_PERIODS = [20, 50]
RSI_PERIOD = 14
BBANDS_PERIOD = 20
ATR_PERIOD = 14
