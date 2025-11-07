import os
import logging
import csv
from datetime import datetime

# -------------------------------------------------------------------
# Global logger setup
# -------------------------------------------------------------------
def setup_logger(name="training", log_dir="logging", level=logging.INFO):
    """
    Sets up a global logger for both console and file output.
    Creates a new log file under `logging/debug_logs.txt`.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "debug_logs.txt"))
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Logger initialized for {name}")
    return logger


# -------------------------------------------------------------------
# CSV utilities for tracking metrics
# -------------------------------------------------------------------
def append_csv(file_path, row_dict):
    """
    Append a single row (dict) to CSV file.
    If file doesn't exist, creates headers automatically.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    write_header = not os.path.exists(file_path)
    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def log_info(message, log_dir="logging"):
    """
    Also write quick info message to training log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "debug_logs.txt")
    with open(path, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)
