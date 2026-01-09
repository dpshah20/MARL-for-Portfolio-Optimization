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
    if scales is None:
        scales = DEFAULT_SCALES

    # 1. Normalize components (add epsilon to avoid division by zero)
    # ret is "Good", Vol/CVaR/MDD are "Bad" (Risks)
    ret = components.get("ret", 0.0) / (scales["ret_scale"] + 1e-8)
    vol = components.get("vol", 0.0) / (scales["vol_scale"] + 1e-8)
    cvar = components.get("cvar", 0.0) / (scales["cvar_scale"] + 1e-8)
    mdd = components.get("mdd", 0.0) / (scales["mdd_scale"] + 1e-8)

    # 2. Construct the Reward Equation
    # The formula subtracts risk terms, so we treat w_vol, w_cvar, w_mdd as penalties.
    # w_meta is expected to be positive (softmax output).
    # w_meta = [w_ret, w_vol, w_cvar, w_mdd]
    
    term_ret  = w_meta[0] * ret
    term_vol  = w_meta[1] * vol
    term_cvar = w_meta[2] * cvar
    term_mdd  = w_meta[3] * mdd
    
    # Raw Shaped Reward = w_ret*ret - w_vol*vol - w_cvar*cvar - w_mdd*mdd
    raw_reward = term_ret - term_vol - term_cvar - term_mdd

    # 3. Apply Cash Exposure Scaling
    # If rho=0.9 (90% cash), exposure=0.1. The agent only "feels" 10% of the reward.
    # This prevents the agent from learning risky behavior when it should be sitting out.
    exposure = 1.0 - rho
    final_reward = exposure * raw_reward

    # 4. Clip for numerical stability
    # Prevents massive outliers from destabilizing the Critic network
    Rmax = scales.get("Rmax", 10.0)
    return float(np.clip(final_reward, -Rmax, Rmax))