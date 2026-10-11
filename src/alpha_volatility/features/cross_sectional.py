"""
Cross-Sectional Features Module
Point-in-time cross-sectional ranking across assets available at each timestamp.
Guarantees no future survivorship or forward-looking universe selection.
"""

from typing import List

import pandas as pd


def compute_cross_sectional_features(
    df: pd.DataFrame,
    features_to_rank: List[str] = None,
) -> pd.DataFrame:
    """
    Computes cross-sectional ranks for specified features at each timestamp t.

    For each timestamp t:
      rank_pct = (rank(x) - 1) / (N_t - 1)  (if N_t > 1, mapped to [-0.5, 0.5])
      if N_t == 1, rank_pct = 0.0 (neutral).

    Args:
        df: DataFrame containing 'timestamp', 'asset_id', and target features.
        features_to_rank: List of feature names to rank cross-sectionally.

    Returns:
        DataFrame with cross-sectional rank features added.
    """
    if features_to_rank is None:
        features_to_rank = ["log_return_1", "momentum_6", "realized_vol_12"]
    out = df.copy()

    for feat in features_to_rank:
        if feat not in out.columns:
            continue

        rank_col = f"cs_rank_{feat}"
        # Group by timestamp strictly point-in-time
        def rank_group(group: pd.Series) -> pd.Series:
            n = len(group)
            if n <= 1:
                return pd.Series(0.0, index=group.index)
            # Ranks from 1 to n
            r = group.rank(method="average", ascending=True)
            # Centered percentile rank [-0.5, +0.5]
            return (r - 1.0) / (n - 1.0) - 0.5

        out[rank_col] = out.groupby("timestamp")[feat].transform(rank_group)

    return out
