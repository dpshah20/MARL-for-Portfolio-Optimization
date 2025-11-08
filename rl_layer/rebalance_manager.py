# rl_layer/rebalance_manager.py
import math
from typing import Dict, Tuple, List

def allocations_to_shares(proposed_alloc: Dict[str, float],
                          current_alloc: Dict[str, float],
                          prices: Dict[str, float],
                          NAV: float,
                          per_asset_delta_thresh: float = 0.01,
                          turnover_thresh: float = 0.02,
                          min_trade_value: float = 1000.0,
                          cap: float = 0.2,
                          min_cash: float = 0.05,
                          lot_size: int = 1) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Convert desired allocations -> executed share counts and executed allocation weights.
    - proposed_alloc: desired weights (0..1)
    - current_alloc: current allocation weights
    - prices: dict ticker->price
    - NAV: current nav
    Returns (shares_to_trade_dict, executed_allocations)
    Note: this is a high-level wrapper — actual execution should be done by Portfolio.execute_allocations.
    """
    # Apply per-asset cap
    capped = {t: min(proposed_alloc.get(t, 0.0), cap) for t in proposed_alloc}

    # total desired
    total = sum(capped.values())
    if total > 1.0:
        # scale down to 1.0
        factor = 1.0 / total
        for t in capped:
            capped[t] *= factor

    # compute desired shares
    desired_shares = {}
    for t, w in capped.items():
        price = float(prices.get(t, 0.0)) or 0.0
        if price <= 0:
            desired_shares[t] = 0
            continue
        value = w * NAV * (1.0 - min_cash)
        shares = math.floor(value / price / lot_size) * lot_size
        desired_shares[t] = max(0, int(shares))

    # compute executed_alloc approximate
    executed_alloc = {}
    mv = sum(desired_shares[t] * float(prices.get(t, 0.0) or 0.0) for t in desired_shares)
    total_nav = NAV if NAV > 0 else 1.0
    for t in desired_shares:
        executed_alloc[t] = (desired_shares[t] * float(prices.get(t, 0.0) or 0.0)) / total_nav if total_nav > 0 else 0.0

    return desired_shares, executed_alloc


class HysteresisSelector:
    """
    Simple top-k selector with hysteresis (prevents rapid churning).
    - keep `k` names selected, allow replacements only after hysteresis_days pass
    """
    def __init__(self, tickers: List[str], k: int = 10, hysteresis_days: int = 3):
        self.tickers = list(tickers)
        self.k = k
        self.hysteresis_days = hysteresis_days
        self.selected = []  # current selected tickers in order
        self.days_since_change = 0

    def update(self, ranked_list: List[str]):
        # ranked_list: from best to worst
        topk = ranked_list[:self.k]
        if not self.selected:
            self.selected = topk.copy()
            self.days_since_change = 0
            return self.selected

        # if topk equals selected, increment day counter
        if topk == self.selected:
            self.days_since_change += 1
            return self.selected

        # if hysteresis not passed, do not change unless large difference
        if self.days_since_change < self.hysteresis_days:
            # compute Jaccard similarity; if low (<0.5), force change
            inter = len(set(topk).intersection(set(self.selected)))
            sim = inter / float(self.k)
            if sim > 0.5:
                self.days_since_change += 1
                return self.selected

        # otherwise adopt new topk
        self.selected = topk.copy()
        self.days_since_change = 0
        return self.selected
