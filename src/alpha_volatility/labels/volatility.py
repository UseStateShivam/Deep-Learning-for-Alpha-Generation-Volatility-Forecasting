"""
Future Realized Volatility Label Generation
Computes forward realized volatility over horizon [t, t+h].
"""

import numpy as np
import pandas as pd


def compute_future_realized_volatility(
    df: pd.DataFrame,
    horizon_bars: int = 3,
    price_col: str = "close",
    target_col: str = "target_realized_vol",
    log_target_col: str = "target_log_realized_vol",
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Computes forward realized volatility:
      RV(t, h) = sqrt(sum_{i=1}^h r(t+i, 1)^2)
      where r(t+i, 1) = ln(P(t+i) / P(t+i-1)).

    Also provides log-transformed target: ln(RV(t, h) + epsilon).

    Args:
        df: DataFrame with ['timestamp', 'asset_id', price_col].
        horizon_bars: Forecast horizon in number of forward bars.
        price_col: Column name of price.
        target_col: Realized volatility column name.
        log_target_col: Log realized volatility column name.
        epsilon: Small positive constant for numerical safety.

    Returns:
        DataFrame with volatility target columns appended.
    """
    out_dfs = []
    for _, asset_df in df.groupby("asset_id", as_index=False):
        adf = asset_df.sort_values("timestamp").copy()
        log_prices = np.log(adf[price_col].values)

        n = len(log_prices)
        rv_vals = np.full(n, fill_value=np.nan)
        log_rv_vals = np.full(n, fill_value=np.nan)

        if n > horizon_bars:
            # 1-bar log returns
            one_bar_rets = np.zeros(n)
            one_bar_rets[1:] = log_prices[1:] - log_prices[:-1]
            sq_rets = one_bar_rets ** 2

            for t in range(n - horizon_bars):
                # Returns from t+1 to t+h
                # In indexing, sq_rets[t+1 : t+h+1] contains the h squared forward returns
                forward_sq = sq_rets[t + 1 : t + horizon_bars + 1]
                rv = np.sqrt(np.sum(forward_sq))
                rv_vals[t] = rv
                log_rv_vals[t] = np.log(rv + epsilon)

        adf[target_col] = rv_vals
        adf[log_target_col] = log_rv_vals
        out_dfs.append(adf)

    result = pd.concat(out_dfs, ignore_index=True)
    return result
