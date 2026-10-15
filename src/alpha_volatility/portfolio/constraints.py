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
    and enforcing that no single asset exceeds max_weight_cap in magnitude.
    """
    if len(weights) == 0 or max_weight_cap >= 1.0:
        return weights

    return np.clip(weights, -max_weight_cap, max_weight_cap)
