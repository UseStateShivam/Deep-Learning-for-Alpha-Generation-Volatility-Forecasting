"""
Statistical Tests and Uncertainty Analysis
Implements Stationary Block Bootstrap for serially dependent time-series metrics
and Diebold-Mariano paired forecast comparison.
"""

from typing import Dict, Tuple

import numpy as np
import scipy.stats as stats


def stationary_block_bootstrap(
    series: np.ndarray,
    block_size: int = 78,
    num_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Performs block bootstrap resampling to estimate standard errors and confidence intervals
    in the presence of serial dependence / autocorrelation.

    Args:
        series: 1D array of metric observations (e.g. timestamp-level ICs).
        block_size: Average block length (e.g. 78 bars = 1 trading day).
        num_bootstraps: Number of bootstrap replications.
        confidence_level: Desired coverage probability (default 0.95).
        seed: Random seed.

    Returns:
        Dictionary containing mean, standard error, and lower/upper bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    if n < block_size:
        return {
            "mean": float(np.mean(series)),
            "std_error": float(np.std(series, ddof=1)),
            "ci_lower": float(np.quantile(series, (1 - confidence_level) / 2)),
            "ci_upper": float(np.quantile(series, 1 - (1 - confidence_level) / 2)),
        }

    # Number of full blocks to tile length n
    num_blocks = int(np.ceil(n / block_size))
    replicates = np.zeros(num_bootstraps)

    for b in range(num_bootstraps):
        # Sample starting block indices uniformly
        start_indices = rng.integers(0, n - block_size + 1, size=num_blocks)
        sampled_blocks = [series[idx : idx + block_size] for idx in start_indices]
        sample = np.concatenate(sampled_blocks)[:n]
        replicates[b] = np.mean(sample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.quantile(replicates, alpha / 2.0))
    ci_upper = float(np.quantile(replicates, 1.0 - alpha / 2.0))

    return {
        "mean": float(np.mean(series)),
        "bootstrap_std_error": float(np.std(replicates)),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def diebold_mariano_test(
    loss_model1: np.ndarray,
    loss_model2: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Computes the Diebold-Mariano test statistic for comparing forecast accuracy.
    H0: Both models have equal forecast accuracy (mean loss differential d = 0).

    Args:
        loss_model1: Loss series of model 1.
        loss_model2: Loss series of model 2.
        h: Forecast horizon (lags for autocovariance truncation).

    Returns:
        (DM_statistic, p_value)
    """
    d = loss_model1 - loss_model2
    n = len(d)
    mean_d = np.mean(d)

    # Sample autocovariance up to lag h - 1
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        gamma_sum += 2.0 * gamma_k

    lr_var = (gamma_0 + gamma_sum) / n
    if lr_var <= 1e-12:
        return 0.0, 1.0

    dm_stat = float(mean_d / np.sqrt(lr_var))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_val
