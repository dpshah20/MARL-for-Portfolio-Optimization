# rl_layer/reward_function.py
"""
Reward scalarization: linear combination of metrics using meta weights.
"""

import numpy as np

DEFAULT_SCALES = {
    "ret_scale": 0.01,
    "vol_scale": 0.02,
    "cvar_scale": 0.02,
    "mdd_scale": 0.2,
    "Rmax": 50.0
}

def compute_reward(components: dict, w_meta: np.ndarray, scales: dict = None):
    if scales is None:
        scales = DEFAULT_SCALES
    ret = components.get("ret", 0.0) / scales["ret_scale"]
    vol = components.get("vol", 0.0) / scales["vol_scale"]
    cvar = components.get("cvar", 0.0) / scales["cvar_scale"]
    mdd = components.get("mdd", 0.0) / scales["mdd_scale"]
    vec = np.array([ret, -vol, -cvar, -mdd])
    raw = float(np.dot(w_meta, vec))
    Rmax = scales.get("Rmax", 50.0)
    return max(-Rmax, min(Rmax, raw))
