import numpy as np


def compute_reward_details(components: dict, w_meta: np.ndarray, rho: float, scales: dict) -> dict:
    """
    Computes a detailed reward breakdown for diagnostics.
    scales must contain: ret_scale, vol_scale, cvar_scale, mdd_scale, Rmax.
    Loaded from configs/params.yaml reward_scales block.
    """

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

def compute_reward(components: dict, w_meta: np.ndarray, rho: float, scales: dict) -> float:
    """
    Computes the risk-aware reward shaped by the Meta-Agent.
    Formula: Reward = (1 - rho) * [ w_ret*ret - w_vol*vol - w_cvar*cvar - w_mdd*mdd ]
    scales must contain: ret_scale, vol_scale, cvar_scale, mdd_scale, Rmax.
    """
    details = compute_reward_details(components, w_meta, rho, scales)
    return details["clipped_reward"]