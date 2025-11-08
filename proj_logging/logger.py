# proj_logging/logger.py
import os
import csv
import json
import logging

# --------------------------------------------------------------------- #
# Logger setup
# --------------------------------------------------------------------- #
def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup standard project logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    logger.info(f"Logger initialized for {name}")
    return logger

# --------------------------------------------------------------------- #
# Append CSV row
# --------------------------------------------------------------------- #
def append_csv(path: str, row: dict):
    """Append dictionary as one row in a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# --------------------------------------------------------------------- #
# Append JSONL (new)
# --------------------------------------------------------------------- #
def append_jsonl(path: str, record: dict):
    """Append JSON record as a single line (for execution logs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

# --------------------------------------------------------------------- #
# Simple log message helper
# --------------------------------------------------------------------- #
def log_info(message: str):
    print(message)
