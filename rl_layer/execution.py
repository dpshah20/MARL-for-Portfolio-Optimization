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

    def __init__(self, tickers, initial_nav: float = 1000.0, initial_cash: float = 1_00_00_000.0, commission_rate: float = 0.001):
        self.tickers = list(tickers)
        self.holdings = {t: 0 for t in self.tickers}
        self.avg_cost = {t: 0.0 for t in self.tickers}
        self.allocations = {t: 0.0 for t in self.tickers}
        
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.commission_rate = float(commission_rate)
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
        # If target_cash = 0.50 (50%), we must hold 50% in cash.
        # So we can only invest the remaining 50%.
        investable_capital = self.total_value * (1.0 - target_cash)
        
        # 3. Calculate Target Rupee Value per Stock
        # target_weights from Actor usually sum to ~1.0 (Softmax).
        # We map that 1.0 to the "Investable Capital" portion only.
        max_value_per_stock = max_weight * self.total_value
        desired_values = {}
        for t in self.tickers:
            w = target_weights.get(t, 0.0)
            desired_values[t] = min(w * investable_capital, max_value_per_stock)

        # 4. Calculate Shares to Buy/Sell
        trades_diff = {}
        for t in self.tickers:
            price = float(prices.get(t, 0.0)) or 0.0
            if price <= 0: 
                continue
            
            current_shares = self.holdings.get(t, 0)
            # Floor to lot size
            target_shares = math.floor(desired_values[t] / price / lot_size) * lot_size
            
            delta = int(target_shares - current_shares)
            
            # Filter tiny trades to save commissions/noise
            if abs(delta * price) >= min_trade_value:
                trades_diff[t] = delta

        # 5. Execute Trades
        # Note: We process all trades, but practically sells (negative delta) add cash immediately,
        # allowing buys (positive delta) to proceed.
        trades_list = []
        
        for t, delta in trades_diff.items():
            price = float(prices.get(t, 0.0))
            if price <= 0: continue

            current_shares = self.holdings.get(t, 0)
            avg_cost_before = float(self.avg_cost.get(t, 0.0))
            cost = delta * price
            commission = abs(cost) * self.commission_rate
            total_cost = cost + commission
            realized_pnl = 0.0
            
            # Safety check: Do we have cash for a BUY?
            if delta > 0 and total_cost > self.cash:
                # Skip buy if insufficient funds 
                # (In a real system, we'd scale down, here we skip for safety)
                continue

            if delta < 0:
                sell_qty = abs(delta)
                effective_sell_price = price
                realized_pnl = (effective_sell_price - avg_cost_before) * sell_qty - commission
                
            self.holdings[t] += delta
            self.cash -= total_cost

            if delta > 0:
                buy_qty = delta
                prev_qty = current_shares
                new_qty = self.holdings[t]
                if new_qty > 0:
                    gross_prev_cost = prev_qty * avg_cost_before
                    gross_new_cost = buy_qty * price + commission
                    self.avg_cost[t] = (gross_prev_cost + gross_new_cost) / new_qty
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
                "comm": commission,
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