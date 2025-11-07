# training/evaluator.py
import numpy as np

def sharpe(returns, rf=0.0):
    mean = np.mean(returns) - rf
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return mean / std * np.sqrt(252)

def max_drawdown(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return dd.min()

def cvar(returns, alpha=0.05):
    r = np.sort(returns)
    k = int(np.ceil(alpha * len(r)))
    if k==0:
        return r.mean()
    return r[:k].mean()
