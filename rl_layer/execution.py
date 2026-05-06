import math
import os
from typing import Dict, Tuple, List
from datetime import datetime

class Portfolio:
    """
    Portfolio bookkeeping & execution manager.
    Tracks:
      - holdings: shares per ticker
      - allocations: portfolio weights
      - cash: uninvested funds (actual rupees)
      - nav: normalized NAV (starting 1000)
    """

    def __init__(self, tickers, initial_nav: float = 1000.0, initial_cash: float = 1_00_00_000.0, commission_rate: float = 0.0):
        self.tickers = list(tickers)
        self.holdings = {t: 0 for t in self.tickers}
        self.avg_cost = {t: 0.0 for t in self.tickers}
        self.allocations = {t: 0.0 for t in self.tickers}
        
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.nav = float(initial_nav)
        self.total_value = float(initial_cash)

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
        self.total_value = self.cash + mv
        self.nav = 1000.0 * (self.total_value / self.initial_cash)
        return self.nav

    # --------------------------------------------------------------------- #
    def execute_allocations(self,
                            target_weights: Dict[str, float],
                            prices: Dict[str, float],
                            target_cash: float,  # <--- CHANGED: Dynamic Rho from Meta-Agent
                            lot_size: int = 1,
                            min_trade_value: float = 1000.0,
                            max_weight: float = 0.20) -> Tuple[Dict[str, float], List[dict]]:
        """
        Execute trades to reach target_weights, STRICTLY respecting target_cash.
        
        Args:
            target_weights: Desired weights for STOCKS (sum should be <= 1.0)
            prices: Current market prices
            target_cash: Fraction of NAV to hold in cash (e.g., 0.1 to 0.9)
            lot_size: Minimum share lot size
            min_trade_value: Minimum value to trigger a trade
            
        Returns:
            executed_allocs: The actual resulting weights
            trades_list: List of dictionaries describing executed trades
        """
        # 1. Calculate Total Capital (Cash + Stock Value)
        market_val = self._market_value(prices)
        self.total_value = market_val + self.cash

        # 2. Determine Investable Portion
        investable_capital = self.total_value * (1.0 - target_cash)
        
        # 3. Calculate Target Rupee Value per Stock
        max_value_per_stock = max_weight * self.total_value
        desired_values = {}
        for t in self.tickers:
            w = target_weights.get(t, 0.0)
            desired_values[t] = min(w * investable_capital, max_value_per_stock)

        # 4. Calculate delta shares; apply rebalancing threshold (Fix #5)
        trades_diff = {}
        for t in self.tickers:
            price = float(prices.get(t, 0.0)) or 0.0
            if price <= 0:
                continue

            current_shares = self.holdings.get(t, 0)
            target_shares = math.floor(desired_values[t] / price / lot_size) * lot_size
            delta = int(target_shares - current_shares)

            # Skip tiny rupee moves
            if abs(delta * price) < min_trade_value:
                continue

            # Skip if weight change is below 5% threshold (Fix #5)
            current_weight = (current_shares * price) / self.total_value if self.total_value > 0 else 0.0
            target_weight = desired_values[t] / self.total_value if self.total_value > 0 else 0.0
            if abs(target_weight - current_weight) < 0.05:
                continue

            trades_diff[t] = delta

        # 5. Execute Trades — sells first, then buys (Fix #3)
        # Processing sells first frees cash so buys can draw on it.
        trades_list = []
        ordered_trades = (
            [(t, d) for t, d in trades_diff.items() if d < 0] +
            [(t, d) for t, d in trades_diff.items() if d > 0]
        )

        for t, delta in ordered_trades:
            price = float(prices.get(t, 0.0))
            if price <= 0:
                continue

            current_shares = self.holdings.get(t, 0)
            avg_cost_before = float(self.avg_cost.get(t, 0.0))
            cost = delta * price   # negative for sells
            commission = 0.0       # commission removed (Fix #6)
            total_cost = cost      # no commission

            # For buys: scale down proportionally if cash is insufficient (Fix #4)
            if delta > 0 and total_cost > self.cash:
                affordable_shares = math.floor(self.cash / price / lot_size) * lot_size
                if affordable_shares <= 0:
                    continue
                delta = affordable_shares
                cost = delta * price
                total_cost = cost

            realized_pnl = 0.0
            if delta < 0:
                realized_pnl = (price - avg_cost_before) * abs(delta)

            self.holdings[t] += delta
            self.cash -= total_cost

            if delta > 0:
                prev_qty = current_shares
                new_qty = self.holdings[t]
                if new_qty > 0:
                    self.avg_cost[t] = (prev_qty * avg_cost_before + delta * price) / new_qty
            elif self.holdings[t] <= 0:
                self.avg_cost[t] = 0.0

            trades_list.append({
                "ticker": t,
                "action": "BUY" if delta > 0 else "SELL",
                "shares": delta,
                "buy_shares": delta if delta > 0 else 0,
                "sell_shares": abs(delta) if delta < 0 else 0,
                "price": price,
                "value": cost,
                "comm": 0.0,
                "avg_cost_before": avg_cost_before,
                "avg_cost_after": float(self.avg_cost.get(t, 0.0)),
                "realized_pnl": realized_pnl,
                "position_after": int(self.holdings[t]),
            })

        # 6. Update Final Stats
        new_mv = self._market_value(prices)
        self.total_value = new_mv + self.cash
        # NAV normalized to initial 1000
        self.nav = 1000.0 * (self.total_value / self.initial_cash)
        
        # Calculate actual executed weights (for logging)
        executed_allocs = {}
        for t in self.tickers:
            val = self.holdings[t] * prices.get(t, 0)
            executed_allocs[t] = val / self.total_value if self.total_value > 0 else 0.0
            self.allocations[t] = executed_allocs[t]

        # Return actual executed weights and trade list for the central logger
        return executed_allocs, trades_list

    # --------------------------------------------------------------------- #
    def apply_open_to_open(self,
                           prev_prices: Dict[str, float],
                           next_prices: Dict[str, float]) -> float:
        """
        Compute open-to-open portfolio return based on price changes.
        Update NAV based on the new prices (next_prices).
        """
        # Value at start (using previous prices)
        prev_total = self.cash + self._market_value(prev_prices)
        
        # Value at end (using next prices) - Holdings haven't changed yet
        next_total = self.cash + self._market_value(next_prices)
        
        # Return calculation
        ret = (next_total / prev_total - 1.0) if prev_total > 0 else 0.0
        
        # Update normalized NAV
        self.total_value = next_total
        self.nav = 1000.0 * (self.total_value / self.initial_cash)
        
        # Removed internal logging - handled by Trainer
        return ret