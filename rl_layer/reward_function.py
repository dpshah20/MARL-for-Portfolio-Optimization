import numpy as np

# Default scaling factors to normalize raw metrics
# These help bring different units (Returns vs Volatility) into a similar numerical range
DEFAULT_SCALES = {
    "ret_scale": 0.01,
    "vol_scale": 0.02,
    "cvar_scale": 0.05,
    "mdd_scale": 0.10,
    "Rmax": 10.0  # Clipping threshold to prevent exploding gradients
}


def compute_reward_details(components: dict, w_meta: np.ndarray, rho: float, scales: dict = None) -> dict:
    """
    Computes a detailed reward breakdown for diagnostics.
    """
    if scales is None:
        scales = DEFAULT_SCALES

    # Raw components
    ret_raw = float(components.get("ret", 0.0))
    vol_raw = float(components.get("vol", 0.0))
    cvar_raw = float(components.get("cvar", 0.0))
    mdd_raw = float(components.get("mdd", 0.0))

    # Normalized components
    ret = ret_raw / (scales["ret_scale"] + 1e-8)
    vol = vol_raw / (scales["vol_scale"] + 1e-8)
    cvar = cvar_raw / (scales["cvar_scale"] + 1e-8)
    mdd = mdd_raw / (scales["mdd_scale"] + 1e-8)

    # Contribution terms
    term_ret = float(w_meta[0]) * ret
    term_vol = float(w_meta[1]) * vol
    term_cvar = float(w_meta[2]) * cvar
    term_mdd = float(w_meta[3]) * mdd

    raw_reward = term_ret - term_vol - term_cvar - term_mdd

    exposure = 1.0 - float(rho)
    final_reward = exposure * raw_reward

    rmax = scales.get("Rmax", 10.0)
    clipped_reward = float(np.clip(final_reward, -rmax, rmax))

    return {
        "ret_raw": ret_raw,
        "vol_raw": vol_raw,
        "cvar_raw": cvar_raw,
        "mdd_raw": mdd_raw,
        "ret_norm": float(ret),
        "vol_norm": float(vol),
        "cvar_norm": float(cvar),
        "mdd_norm": float(mdd),
        "term_ret": float(term_ret),
        "term_vol": float(term_vol),
        "term_cvar": float(term_cvar),
        "term_mdd": float(term_mdd),
        "raw_reward": float(raw_reward),
        "exposure": float(exposure),
        "final_reward": float(final_reward),
        "clipped_reward": float(clipped_reward),
    }

def compute_reward(components: dict, w_meta: np.ndarray, rho: float, scales: dict = None) -> float:
    """
    Computes the risk-aware reward shaped by the Meta-Agent.
    
    Formula: Reward = (1 - rho) * [ w_ret*ret - w_vol*vol - w_cvar*cvar - w_mdd*mdd ]
    
    Args:
        components (dict): Daily metrics {'ret', 'vol', 'cvar', 'mdd'}.
        w_meta (np.ndarray): 4D weight vector [w_ret, w_vol, w_cvar, w_mdd].
        rho (float): Cash fraction (0.1 to 0.9).
        scales (dict): Optional scaling factors to normalize raw metrics.
        
    Returns:
        float: The calculated scalar reward.
    """
    details = compute_reward_details(components, w_meta, rho, scales=scales)
    return details["clipped_reward"]