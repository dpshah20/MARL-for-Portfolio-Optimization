import math
import os
import json
from typing import Dict
from datetime import datetime

# ------------------------------------------------------------------------- #
# Lightweight logging utility
# ------------------------------------------------------------------------- #
def append_jsonl(path: str, record: dict):
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
      - cash: uninvested funds (actual rupees)
      - nav: normalized NAV (starting 1000)
    """

    def __init__(self, tickers, initial_nav: float = 1000.0, initial_cash: float = 1_00_00_000.0, log_dir: str = "logs"):
        self.tickers = list(tickers)
        self.holdings = {t: 0 for t in self.tickers}
        self.allocations = {t: 0.0 for t in self.tickers}
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.nav = float(initial_nav)

        # setup execution log path
        self.log_path = os.path.join(log_dir, "execution_layer_logs.jsonl")
        os.makedirs(log_dir, exist_ok=True)

        append_jsonl(self.log_path, {
            "event": "init_portfolio",
            "normalized_nav": self.nav,
            "cash": self.cash,
            "tickers_count": len(self.tickers)
        })

    # --------------------------------------------------------------------- #
    def _market_value(self, prices: Dict[str, float]) -> float:
        """Total rupee market value of all holdings."""
        return sum(self.holdings.get(t, 0) * float(prices.get(t, 0.0)) for t in self.tickers)

    def compute_nav(self, prices: Dict[str, float]) -> float:
        """
        Compute normalized NAV from rupee total portfolio value.
        NAV = 1000 * (current_value / initial_cash)
        """
        mv = self._market_value(prices)
        total_value = self.cash + mv
        self.nav = 1000.0 * (total_value / self.initial_cash)
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
        Uses actual rupee values but logs normalized NAV.
        """
        total_value = self._market_value(prices) + self.cash
        investable_value = max(0.0, total_value * (1.0 - min_cash))

        # Enforce caps and normalization
        tgt_w = {t: min(target_weights.get(t, 0.0), cap_per_asset) for t in self.tickers}
        total_tgt = sum(tgt_w.values())
        if total_tgt > 1.0:
            scale = 1.0 / total_tgt
            for t in tgt_w:
                tgt_w[t] *= scale

        # Determine target shares
        desired_shares = {}
        for t in self.tickers:
            p = float(prices.get(t, 0.0)) or 0.0
            if p <= 0:
                desired_shares[t] = self.holdings.get(t, 0)
                continue
            desired_value = tgt_w[t] * investable_value
            shares = math.floor(desired_value / p / lot_size) * lot_size
            desired_shares[t] = max(0, int(shares))

        # Compute trade deltas
        trades = {t: desired_shares[t] - self.holdings.get(t, 0) for t in self.tickers}

        # Filter out small trades
        final_trades = {}
        for t in self.tickers:
            p = float(prices.get(t, 0.0)) or 0.0
            value = abs(trades[t]) * p
            final_trades[t] = trades[t] if value >= min_trade_value else 0

        # Execute trades
        trade_records = []
        for t, delta in final_trades.items():
            if delta == 0:
                continue
            price = float(prices.get(t, 0.0)) or 0.0
            trade_value = delta * price
            old_cash = self.cash

            # Handle buy/sell cash update
            if trade_value > 0 and trade_value > self.cash:
                # Not enough cash -> scale down
                affordable_shares = math.floor(self.cash / price / lot_size) * lot_size
                delta = affordable_shares
                trade_value = delta * price

            self.holdings[t] += delta
            self.cash -= trade_value

            trade_records.append({
                "ticker": t,
                "price": price,
                "delta_shares": delta,
                "trade_value": trade_value,
                "old_cash": old_cash,
                "new_cash": self.cash
            })

        # Recalculate allocations & NAV
        mv = self._market_value(prices)
        total_value = mv + self.cash
        for t in self.tickers:
            price = float(prices.get(t, 0.0)) or 0.0
            self.allocations[t] = (self.holdings[t] * price) / total_value if total_value > 0 else 0.0

        self.nav = 1000.0 * (total_value / self.initial_cash)

        append_jsonl(self.log_path, {
            "timestamp": date or datetime.now().isoformat(),
            "event": "execute_trades",
            "nav_normalized": self.nav,
            "cash": self.cash,
            "total_value": total_value,
            "market_value": mv,
            "executed_allocations": self.allocations,
            "trades": trade_records
        })

        return dict(self.allocations)

    # --------------------------------------------------------------------- #
    def apply_open_to_open(self,
                           prev_prices: Dict[str, float],
                           next_prices: Dict[str, float],
                           date: str = None) -> float:
        """
        Compute open-to-open portfolio return and update NAV (normalized).
        """
        prev_total = self._market_value(prev_prices) + self.cash
        mv_next = self._market_value(next_prices)
        next_total = self.cash + mv_next
        ret = (next_total / prev_total - 1.0) if prev_total > 0 else 0.0

        # Update normalized NAV
        self.nav = 1000.0 * (next_total / self.initial_cash)

        append_jsonl(self.log_path, {
            "timestamp": date or datetime.now().isoformat(),
            "event": "daily_return",
            "nav_before": 1000.0 * (prev_total / self.initial_cash),
            "nav_after": self.nav,
            "daily_return": ret
        })
        return ret
