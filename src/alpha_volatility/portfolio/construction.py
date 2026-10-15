"""
Portfolio Construction Module
Constructs dollar-neutral long/short portfolios from decile signal ranking.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .constraints import apply_position_limits


def assign_decile_quantiles(
    df: pd.DataFrame,
    signal_col: str = "alpha_signal",
    num_quantiles: int = 10,
    min_universe_size: int = 10,
) -> pd.DataFrame:
    """
    Ranks eligible assets cross-sectionally at each timestamp and assigns integer quantile buckets (1..num_quantiles).

    Args:
        df: DataFrame with ['timestamp', 'asset_id', signal_col].
        signal_col: Alpha signal used for ranking.
        num_quantiles: Number of quantile buckets (default 10 for deciles).
        min_universe_size: Minimum number of assets required to construct valid quantiles.

    Returns:
        DataFrame with 'quantile' column added (integer 1..num_quantiles, or NaN if universe < min_universe_size).
    """
    out = df.copy()

    def rank_to_quantile(group: pd.Series) -> pd.Series:
        n = len(group)
        if n < min_universe_size:
            # Fallback if universe is too small for full deciles
            if n < 2:
                return pd.Series(np.nan, index=group.index)
            # Assign into available ranks
            r = group.rank(method="first", ascending=True)
            return np.ceil(r / n * num_quantiles).astype(int)

        r = group.rank(method="first", ascending=True)
        q = np.ceil(r / n * num_quantiles).astype(int)
        return np.clip(q, 1, num_quantiles)

    out["quantile"] = out.groupby("timestamp")[signal_col].transform(rank_to_quantile)
    return out


def construct_dollar_neutral_weights(
    df_timestamp: pd.DataFrame,
    quantile_col: str = "quantile",
    num_quantiles: int = 10,
    max_weight_cap: Optional[float] = None,
) -> Dict[str, float]:
    """
    Constructs dollar-neutral portfolio weights at a single rebalance timestamp:
      - Long: Decile 10 (top), equal-weighted, sum(w_long) = +0.5
      - Short: Decile 1 (bottom), equal-weighted, sum(w_short) = -0.5
      - Deciles 2..9: weight = 0.0
      - Net exposure = 0.0, Gross exposure = 1.0.

    Args:
        df_timestamp: Slice of DataFrame for a single timestamp.
        quantile_col: Quantile bucket column.
        num_quantiles: Max quantile number (10 for deciles).
        max_weight_cap: Optional position limit cap.

    Returns:
        Dictionary mapping asset_id to portfolio weight.
    """
    weights: Dict[str, float] = {str(a): 0.0 for a in df_timestamp["asset_id"].unique()}

    long_assets = df_timestamp[df_timestamp[quantile_col] == num_quantiles]["asset_id"].tolist()
    short_assets = df_timestamp[df_timestamp[quantile_col] == 1]["asset_id"].tolist()

    if not long_assets or not short_assets:
        return weights

    # Equal weight allocation: sum of longs = +0.5, sum of shorts = -0.5
    long_w = 0.5 / len(long_assets)
    short_w = -0.5 / len(short_assets)

    for a in long_assets:
        weights[str(a)] = long_w
    for a in short_assets:
        weights[str(a)] = short_w

    if max_weight_cap is not None and max_weight_cap < 1.0:
        asset_keys = list(weights.keys())
        w_arr = np.array([weights[k] for k in asset_keys])
        w_capped = apply_position_limits(w_arr, max_weight_cap=max_weight_cap)
        return {k: float(w) for k, w in zip(asset_keys, w_capped, strict=False)}

    return weights
