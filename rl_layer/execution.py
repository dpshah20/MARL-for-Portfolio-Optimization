import math
import os
import json
from typing import Dict
from datetime import datetime

# ------------------------------------------------------------------------- #
# Utility logging helper (kept lightweight to avoid dependency issues)
# ------------------------------------------------------------------------- #
def append_jsonl(path: str, record: dict):
    """Append a JSON record as a single line to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

# ------------------------------------------------------------------------- #
# Portfolio Execution Class
# ------------------------------------------------------------------------- #
class Portfolio:
    """
    Portfolio bookkeeping & execution manager.
    Tracks:
      - holdings: shares per ticker
      - allocations: portfolio weights
      - cash: uninvested funds
      - nav: total portfolio value (cash + market value)
    """

    def __init__(self, tickers, initial_nav: float = 1.0, log_dir: str = "logs"):
        self.tickers = list(tickers)
        self.holdings = {t: 0 for t in self.tickers}
        self.allocations = {t: 0.0 for t in self.tickers}
        self.cash = initial_nav
        self.nav = float(initial_nav)

        # setup execution log path
        self.log_path = os.path.join(log_dir, "execution_layer_logs.jsonl")
        os.makedirs(log_dir, exist_ok=True)

    # --------------------------------------------------------------------- #
    def _market_value(self, prices: Dict[str, float]) -> float:
        return sum(self.holdings.get(t, 0) * float(prices.get(t, 0.0)) for t in self.tickers)

    def compute_nav(self, prices: Dict[str, float]) -> float:
        mv = self._market_value(prices)
        self.nav = float(self.cash + mv)
        return self.nav

    # --------------------------------------------------------------------- #
    def execute_allocations(self,
                            target_weights: Dict[str, float],
                            prices: Dict[str, float],
                            lot_size: int = 1,
                            min_trade_value: float = 0.0,
                            cap_per_asset: float = 1.0,
                            min_cash: float = 0.0,
                            date: str = None) -> Dict[str, float]:
        """
        Execute trades to reach target_weights.
        Logs trade actions and post-trade portfolio state.
        """
        current_nav = self.compute_nav(prices)
        investable_nav = max(0.0, current_nav * (1.0 - min_cash))
        tgt_w = {t: min(target_weights.get(t, 0.0), cap_per_asset) for t in self.tickers}

        total_tgt = sum(tgt_w.values())
        if total_tgt > 1.0:
            scale = 1.0 / total_tgt
            for t in tgt_w:
                tgt_w[t] *= scale

        # Desired shares based on prices
        desired_shares = {}
        for t in self.tickers:
            p = float(prices.get(t, 0.0)) or 0.0
            if p <= 0:
                desired_shares[t] = self.holdings.get(t, 0)
                continue
            val = tgt_w[t] * investable_nav
            shares = math.floor(val / p / lot_size) * lot_size
            desired_shares[t] = max(0, int(shares))

        # Compute deltas
        trades = {}
        for t in self.tickers:
            trades[t] = desired_shares[t] - self.holdings.get(t, 0)

        # Skip small-value trades
        final_trades = {}
        for t in self.tickers:
            p = float(prices.get(t, 0.0)) or 0.0
            val = abs(trades[t]) * p
            final_trades[t] = trades[t] if val >= min_trade_value else 0

        # Execute trades
        trade_records = []
        for t, delta in final_trades.items():
            if delta == 0:
                continue
            p = float(prices.get(t, 0.0)) or 0.0
            cost = delta * p
            old_cash = self.cash
            old_hold = self.holdings.get(t, 0)

            # Ensure sufficient cash for buys
            if cost > 0 and cost > self.cash:
                affordable_shares = math.floor(self.cash / p / lot_size) * lot_size
                delta = affordable_shares - old_hold
                cost = delta * p

            self.holdings[t] += delta
            self.cash -= cost

            trade_records.append({
                "ticker": t,
                "price": p,
                "delta_shares": delta,
                "trade_value": cost,
                "old_cash": old_cash,
                "new_cash": self.cash,
                "old_holdings": old_hold,
                "new_holdings": self.holdings[t]
            })

        # Recalculate allocations
        mv = self._market_value(prices)
        total_nav = mv + self.cash
        for t in self.tickers:
            p = float(prices.get(t, 0.0)) or 0.0
            self.allocations[t] = (self.holdings[t] * p) / total_nav if total_nav > 0 else 0.0
        self.nav = total_nav

        # === Log execution === #
        record = {
            "timestamp": date or datetime.now().isoformat(),
            "nav": self.nav,
            "cash": self.cash,
            "prices": prices,
            "target_weights": tgt_w,
            "executed_allocations": self.allocations,
            "trades": trade_records
        }
        append_jsonl(self.log_path, record)

        return dict(self.allocations)

    # --------------------------------------------------------------------- #
    def apply_open_to_open(self,
                           prev_prices: Dict[str, float],
                           next_prices: Dict[str, float],
                           date: str = None) -> float:
        """
        Compute open-to-open portfolio return & update NAV.
        """
        prev_nav = self.compute_nav(prev_prices)
        mv_next = sum(self.holdings[t] * float(next_prices.get(t, 0.0)) for t in self.tickers)
        next_nav = float(self.cash + mv_next)
        self.nav = next_nav

        ret = (next_nav / prev_nav - 1.0) if prev_nav > 0 else 0.0

        # Log daily NAV change
        append_jsonl(self.log_path, {
            "timestamp": date or datetime.now().isoformat(),
            "event": "daily_return",
            "nav_before": prev_nav,
            "nav_after": next_nav,
            "daily_return": ret
        })
        return ret
