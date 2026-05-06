import os
import csv
import logging
from datetime import datetime
import json

class RunLogger:
    def __init__(self, base_dir="logs"):
        """
        Structured Logger for Meta-RL System.
        Creates a unique timestamped folder for every training run.
        
        Directory Structure:
        logs/run_YYYYMMDD_HHMMSS/
          ├── 1_meta_strategy.csv      (Weekly Brain decisions)
          ├── 2_daily_performance.csv  (Daily Body performance)
          ├── 3_trade_history.csv      (Atomic actions)
          ├── 4_training_internals.csv (Gradients & Loss)
          ├── execution_logs.jsonl     (Full execution details)
          └── system.log               (Debug info)
        """
        # 1. Create Unique Run Directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{timestamp}")
        self.stock_dir = os.path.join(self.run_dir, "stocks")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.stock_dir, exist_ok=True)

        # 2. Define Log File Paths
        self.paths = {
            "meta": os.path.join(self.run_dir, "1_meta_strategy.csv"),
            "perf": os.path.join(self.run_dir, "2_daily_performance.csv"),
            "trades": os.path.join(self.run_dir, "3_trade_history.csv"),
            "internal": os.path.join(self.run_dir, "4_training_internals.csv"),
            "execution": os.path.join(self.run_dir, "execution_logs.jsonl"),
            "system": os.path.join(self.run_dir, "system.log")
        }

        # 3. Setup System Logger (Console + File)
        self.sys_logger = logging.getLogger("TRAINER")
        self.sys_logger.setLevel(logging.INFO)
        self.sys_logger.handlers = [] # Clear previous handlers
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        
        # File Handler
        fh = logging.FileHandler(self.paths["system"])
        fh.setFormatter(formatter)
        self.sys_logger.addHandler(fh)
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.sys_logger.addHandler(ch)

        # 4. Initialize CSV Headers
        
        # Meta: What the Brain decided for the week
        self._init_csv("meta", [
            "week_start_date", "step", "phase",
            "rho_cash", "w_ret", "w_vol", "w_cvar", "w_mdd", 
            "log_prob", "entropy", "weekly_return", "meta_loss", "rho_clipped"
        ])
        
        # Perf: How the Body performed daily
        self._init_csv("perf", [
            "date", "step", "phase",
            "nav", "cash", "cash_pct", 
            "daily_return", "reward", 
            "vol_30d", "cvar_30d", "mdd_30d",
            "term_ret", "term_vol", "term_cvar", "term_mdd", "reward_raw", "exposure"
        ])
        
        # Trades: Atomic trade records
        self._init_csv("trades", [
            "date", "step", "ticker", "action", 
            "shares", "buy_shares", "sell_shares", "price", "value", "commission",
            "avg_cost_before", "avg_cost_after", "realized_pnl", "position_after"
        ])
        
        # Internal: Neural Network Health
        self._init_csv("internal", [
            "step", "phase",
            "actor_loss", "critic_loss", "actor_grad_norm", 
            "weight_mean", "weight_std"
        ])

    def _init_csv(self, key, headers):
        with open(self.paths[key], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _append_csv(self, key, row):
        with open(self.paths[key], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # --- Public Logging Methods ---

    def log_meta(self, data: dict):
        """Log weekly strategy decisions (The Brain)."""
        self._append_csv("meta", [
            data.get("date"), data.get("step"), data.get("phase", "train"),
            f"{data.get('rho', 0):.4f}",
            f"{data.get('w_ret', 0):.4f}", f"{data.get('w_vol', 0):.4f}",
            f"{data.get('w_cvar', 0):.4f}", f"{data.get('w_mdd', 0):.4f}",
            f"{data.get('log_prob', 0):.6f}", f"{data.get('entropy', 0):.6f}",
            f"{data.get('weekly_return', 0):.6f}",
            f"{data.get('meta_loss', 0):.6f}",
            int(bool(data.get("rho_clipped", False))),
        ])

    def log_performance(self, data: dict):
        """Log daily portfolio metrics (The Body)."""
        self._append_csv("perf", [
            data.get("date"), data.get("step"), data.get("phase", "train"),
            f"{data.get('nav', 0):.4f}",
            f"{data.get('cash', 0):.2f}",
            f"{data.get('cash_pct', 0):.4f}",
            f"{data.get('ret', 0):.6f}",
            f"{data.get('reward', 0):.6f}",
            f"{data.get('vol', 0):.6f}",
            f"{data.get('cvar', 0):.6f}",
            f"{data.get('mdd', 0):.6f}",
            f"{data.get('term_ret', 0):.6f}",
            f"{data.get('term_vol', 0):.6f}",
            f"{data.get('term_cvar', 0):.6f}",
            f"{data.get('term_mdd', 0):.6f}",
            f"{data.get('raw_reward', 0):.6f}",
            f"{data.get('exposure', 0):.6f}",
        ])

    def log_trade(self, data: dict):
        """Log atomic trade execution."""
        self._append_csv("trades", [
            data.get("date"), data.get("step"),
            data.get("ticker"), data.get("action"),
            data.get("shares"), data.get("buy_shares", 0), data.get("sell_shares", 0),
            f"{data.get('price', 0):.2f}",
            f"{data.get('value', 0):.2f}", f"{data.get('comm', 0):.2f}",
            f"{data.get('avg_cost_before', 0):.4f}",
            f"{data.get('avg_cost_after', 0):.4f}",
            f"{data.get('realized_pnl', 0):.2f}",
            data.get("position_after", 0),
        ])

    def log_execution(self, data: dict):
        """Log full execution details (JSONL)."""
        with open(self.paths["execution"], 'a') as f:
            f.write(json.dumps(data) + "\n")

    def log_internal(self, data: dict):
        self._append_csv("internal", [
            data.get("step"),
            data.get("phase", "train"),
            f"{data.get('a_loss', 0):.6f}",
            f"{data.get('c_loss', 0):.6f}",
            f"{data.get('actor_grad_norm', 0.0):.6f}",
            f"{data.get('w_mean', 0):.4f}",
            f"{data.get('w_std', 0):.4f}"
        ])


    def info(self, msg):
        self.sys_logger.info(msg)

    def error(self, msg):
        self.sys_logger.error(msg)