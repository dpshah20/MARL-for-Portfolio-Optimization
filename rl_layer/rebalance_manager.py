# rl_layer/rebalance_manager.py
import numpy as np
import math
from typing import List, Dict, Tuple

class HysteresisSelector:
    def __init__(self, tickers: List[str], k: int = 10, hysteresis_days: int = 3):
        self.tickers = tickers
        self.k = k
        self.hysteresis_days = hysteresis_days
        self.counters = {t: 0 for t in tickers}
        self.selected = []

    def update(self, ranked_list: List[str]):
        candidates = set(ranked_list[:self.k])
        for t in self.tickers:
            if t in candidates:
                self.counters[t] += 1
            else:
                self.counters[t] = 0
        new_selected = [t for t, c in self.counters.items() if c >= self.hysteresis_days]
        # preserve order according to ranked_list
        new_selected = sorted(new_selected, key=lambda x: ranked_list.index(x) if x in ranked_list else 999)
        self.selected = new_selected
        return new_selected

def allocations_to_shares(a_proposed: Dict[str, float],
                          a_current: Dict[str, float],
                          prices: Dict[str, float],
                          NAV: float,
                          per_asset_delta_thresh: float = 0.01,
                          turnover_thresh: float = 0.02,
                          min_trade_value: float = 1000.0,
                          cap: float = 0.20,
                          min_cash: float = 0.05,
                          lot_size: int = 1):
    tickers = list(a_proposed.keys())
    deltas = {t: abs(a_proposed[t] - a_current.get(t, 0.0)) for t in tickers}
    sum_deltas = sum(deltas.values())
    if sum_deltas < turnover_thresh:
        # no trade
        shares = {}
        executed_alloc = {}
        for t in tickers:
            p = prices.get(t, 1.0)
            shares[t] = int(math.floor((a_current.get(t,0.0)*NAV)/max(1e-8,p)))
            executed_alloc[t] = (shares[t]*p)/NAV
        return shares, executed_alloc
    # enforce caps and investable
    a = a_proposed.copy()
    investable = max(0.0, 1.0 - min_cash)
    total = sum(a.values())
    if total > 0:
        scale = min(1.0, investable / total)
        for t in a:
            a[t] = min(a[t] * scale, cap)
    # to shares
    shares = {}
    exec_amt = {}
    for t in tickers:
        p = prices.get(t, 0.0)
        amt = a.get(t, 0.0) * NAV
        if p <= 0 or amt < min_trade_value:
            s = 0
        else:
            s = int(math.floor((amt / p) / lot_size) * lot_size)
        shares[t] = s
        exec_amt[t] = s * p
    total_exec = sum(exec_amt.values())
    executed_alloc = {t: exec_amt[t] / max(1e-12, NAV) for t in tickers}
    cash_frac = 1.0 - sum(executed_alloc.values())
    if cash_frac < min_cash:
        # scale down proportionally
        invest_frac = 1.0 - min_cash
        total_exec_frac = sum(executed_alloc.values())
        if total_exec_frac > 0:
            factor = invest_frac / total_exec_frac
            for t in tickers:
                executed_alloc[t] = executed_alloc[t] * factor
            # recompute shares
            for t in tickers:
                p = prices.get(t, 1.0)
                shares[t] = int(math.floor((executed_alloc[t]*NAV)/max(1e-8,p)))
                exec_amt[t] = shares[t]*p
            executed_alloc = {t: exec_amt[t] / max(1e-12, NAV) for t in tickers}
    return shares, executed_alloc
