"""
Execution Timing Module
Models realistic chronological execution latency (e.g. 1-bar delay) to eliminate lookahead execution bias.
"""

import pandas as pd


def apply_execution_lag(
    target_weights_df: pd.DataFrame,
    lag_bars: int = 1,
) -> pd.DataFrame:
    """
    Shifts intended target portfolio weights forward by lag_bars to simulate
    realistic execution latency (signal generated on bar t close -> executed at t + lag_bars).

    Args:
        target_weights_df: DataFrame containing ['timestamp', 'asset_id', 'weight'].
        lag_bars: Number of bars to lag execution.

    Returns:
        DataFrame with lagged executable weights.
    """
    if lag_bars <= 0:
        return target_weights_df.copy()

    # Sort and shift per asset
    out_dfs = []
    for _, asset_df in target_weights_df.groupby("asset_id", as_index=False):
        adf = asset_df.sort_values("timestamp").copy()
        adf["weight"] = adf["weight"].shift(lag_bars).fillna(0.0)
        out_dfs.append(adf)

    result = pd.concat(out_dfs, ignore_index=True)
    return result.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)
