"""Feature pipeline configuration"""

import os
from datetime import datetime

# Resolve all paths from repository root so execution CWD does not matter.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
STOCKS_DIR = os.path.join(REPO_ROOT, "data", "nifty100_data")
PROCESSED_DIR = os.path.join(REPO_ROOT, "nifty100_new")
MACROS_DIR = os.path.join(REPO_ROOT, "data", "macros")

# Date range
START_DATE = datetime(2015, 1, 1)

# Feature engineering params
PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS = ["volume"]
MA_PERIODS = [20, 50]
RSI_PERIOD = 14
BBANDS_PERIOD = 20
ATR_PERIOD = 14
