"""
Portfolio Constraints Module
Enforces universe eligibility and maximum single-position allocation caps.
"""

import numpy as np


def apply_position_limits(
    weights: np.ndarray,
    max_weight_cap: float = 0.20,
) -> np.ndarray:
    """
    Clips individual asset weights to max_weight_cap while preserving sign
    and rescaling to maintain gross exposure.
    """
    if len(weights) == 0 or max_weight_cap >= 1.0:
        return weights

    np.sign(weights)
    abs_w = np.abs(weights)
    clipped = np.minimum(abs_w, max_weight_cap)

    # Renormalize long and short legs separately
    long_mask = weights > 0
    short_mask = weights < 0

    out = np.zeros_like(weights)
    if np.sum(long_mask) > 0:
        long_sum = np.sum(clipped[long_mask])
        if long_sum > 0:
            out[long_mask] = (clipped[long_mask] / long_sum) * 0.5

    if np.sum(short_mask) > 0:
        short_sum = np.sum(clipped[short_mask])
        if short_sum > 0:
            out[short_mask] = -(clipped[short_mask] / short_sum) * 0.5

    return out
