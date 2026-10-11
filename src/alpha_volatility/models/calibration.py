"""
Calibration and Probabilistic Forecasting Diagnostics
Implements distribution metrics, probability of positive return, predictive quantiles,
and realized volatility calibration binning.
"""

from typing import Dict, Tuple

import numpy as np
import scipy.stats as stats


def compute_prob_positive_return(
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
) -> np.ndarray:
    """
    Computes conditional probability P(r > 0) under Student-t distribution:
      P(r > 0) = 1 - F((0 - mu) / sigma; df=nu)

    Args:
        mu: Location parameter array.
        sigma: Scale parameter array.
        nu: Degrees of freedom array.

    Returns:
        Array of probabilities in [0, 1].
    """
    z = (0.0 - mu) / np.maximum(sigma, 1e-6)
    cdf_val = stats.t.cdf(z, df=nu)
    return 1.0 - cdf_val


def compute_predictive_quantiles(
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> Dict[str, np.ndarray]:
    """
    Computes parametric quantiles for conditional Student-t return distribution.
    """
    out = {}
    for q in quantiles:
        # Standard Student-t quantile
        t_q = stats.t.ppf(q, df=nu)
        q_val = mu + sigma * t_q
        out[f"q_{int(q * 100):02d}"] = q_val
    return out


def evaluate_interval_coverage(
    realized_returns: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> float:
    """
    Calculates empirical coverage rate: percentage of realized returns falling inside [q_lower, q_upper].
    """
    inside = (realized_returns >= q_lower) & (realized_returns <= q_upper)
    return float(np.mean(inside))


def compute_volatility_calibration_buckets(
    realized_vol: np.ndarray,
    predicted_vol: np.ndarray,
    num_bins: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Bins predicted volatility into quantiles and computes mean realized volatility per bucket.
    Ideal calibration: monotonic 45-degree relationship.
    """
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(predicted_vol, quantiles)
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    bin_indices = np.digitize(predicted_vol, bin_edges) - 1

    bucket_pred = []
    bucket_realized = []

    for b in range(num_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bucket_pred.append(float(np.mean(predicted_vol[mask])))
            bucket_realized.append(float(np.mean(realized_vol[mask])))
        else:
            bucket_pred.append(np.nan)
            bucket_realized.append(np.nan)

    return {
        "bucket_index": np.arange(1, num_bins + 1),
        "mean_predicted_vol": np.array(bucket_pred),
        "mean_realized_vol": np.array(bucket_realized),
    }
