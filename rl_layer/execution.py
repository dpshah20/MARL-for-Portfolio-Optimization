# rl_layer/execution.py
from typing import Dict
import copy

class Portfolio:
    def __init__(self, tickers, initial_nav: float = 1.0):
        self.tickers = tickers
        self.nav = float(initial_nav)
        self.allocations = {t: 0.0 for t in tickers}
        self.shares = {t: 0 for t in tickers}
        self.cash = 1.0
        self.history = []

    def execute_allocations(self, executed_alloc: Dict[str,float], open_prices: Dict[str,float]):
        # executed_alloc are fractional allocations of NAV
        self.allocations = executed_alloc.copy()
        self.shares = {}
        for t in self.tickers:
            p = open_prices.get(t, None)
            if p is None or p <= 0:
                self.shares[t] = 0
            else:
                dollars = self.allocations.get(t, 0.0) * self.nav
                self.shares[t] = dollars / p  # fractional shares allowed for accounting; use integer shares from rebalance manager in production
        self.cash = 1.0 - sum(self.allocations.values())
        self.history.append({"nav": self.nav, "alloc": dict(self.allocations), "cash": self.cash})

    def apply_open_to_open(self, open_t: Dict[str,float], open_t1: Dict[str,float]) -> float:
        # compute per-asset returns and weighted sum
        port_ret = 0.0
        for t in self.tickers:
            p0 = open_t.get(t, None)
            p1 = open_t1.get(t, None)
            if p0 is None or p1 is None or p0 == 0:
                r = 0.0
            else:
                r = (p1 / p0) - 1.0
            port_ret += self.allocations.get(t, 0.0) * r
        self.nav = self.nav * (1.0 + port_ret)
        self.history.append({"nav_after": self.nav, "port_return": port_ret})
        return port_ret
