"""
Future Return Label Generation
Computes forward log-return targets strictly over the forward horizon [t, t+h].
"""

import numpy as np
import pandas as pd


def compute_future_returns(
    df: pd.DataFrame,
    horizon_bars: int = 3,
    price_col: str = "close",
    target_col: str = "target_return",
) -> pd.DataFrame:
    """
    Computes forward log return r(t, h) = ln(P(t+h) / P(t)).

    For an observation at bar t close:
    Target measures the subsequent price change up to bar t+h close.

    Rows at the end of each asset series where t+h extends beyond available data
    will receive NaN and MUST be dropped or excluded from training.

    Args:
        df: DataFrame with ['timestamp', 'asset_id', price_col].
        horizon_bars: Forecast horizon in number of bars (e.g. 3 bars = 15m).
        price_col: Price column used for return calculation ('close').
        target_col: Name of generated label column.

    Returns:
        DataFrame with target column appended.
    """
    out_dfs = []
    for _, asset_df in df.groupby("asset_id", as_index=False):
        adf = asset_df.sort_values("timestamp").copy()
        log_prices = np.log(adf[price_col].values)

        # Forward return: log_prices[t+h] - log_prices[t]
        fwd_returns = np.full_like(log_prices, fill_value=np.nan)
        if len(log_prices) > horizon_bars:
            fwd_returns[:-horizon_bars] = log_prices[horizon_bars:] - log_prices[:-horizon_bars]

        adf[target_col] = fwd_returns
        out_dfs.append(adf)

    result = pd.concat(out_dfs, ignore_index=True)
    return result
