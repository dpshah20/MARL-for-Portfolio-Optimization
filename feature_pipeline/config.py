# feature_pipeline/config.py
import os

# Paths (edit these to your local paths if needed)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
STOCKS_DIR = os.path.join(DATA_DIR, "ohlcv_data")
MACROS_DIR = os.path.join(DATA_DIR, "macros")
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")  # place processed/ inside project root
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Date filter
START_DATE = "2015-01-01"   # discard data before this date

# Indicator params
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ADX_PERIOD = 14
ATR_PERIOD = 14
BOLL_PERIOD = 20
BOLL_STD = 2

# Windowing for encoders
WINDOW_SIZE = 30

# Graph builder defaults
GRAPH_LOOKBACK = 60
GRAPH_METHOD = "knn"   # "threshold" or "knn"
GRAPH_K = 8
GRAPH_THRESHOLD = 0.6
