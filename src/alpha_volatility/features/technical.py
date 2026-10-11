"""
Technical Features Module
Vectorized, point-in-time calculation of price, volatility, volume, and momentum features.
Ensures zero future information leakage: every feature at time t uses ONLY observations <= t.
"""

import numpy as np
import pandas as pd


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all single-asset technical features grouped by asset_id.

    Args:
        df: DataFrame containing canonical columns ['timestamp', 'asset_id', 'open', 'high', 'low', 'close', 'volume'].

    Returns:
        DataFrame with technical feature columns added.
    """
    out_dfs = []

    # Process per asset to prevent cross-contamination
    for _, asset_df in df.groupby("asset_id", as_index=False):
        adf = asset_df.sort_values("timestamp").copy()

        # Log prices and returns
        close_p = adf["close"].values
        open_p = adf["open"].values
        high_p = adf["high"].values
        low_p = adf["low"].values
        vol_p = adf["volume"].values

        log_close = np.log(close_p)
        log_open = np.log(open_p)
        log_high = np.log(high_p)
        log_low = np.log(low_p)

        # 1. Price and Return features
        ret_1 = np.zeros_like(log_close)
        ret_1[1:] = log_close[1:] - log_close[:-1]
        adf["log_return_1"] = ret_1

        for lag in [3, 6, 12]:
            ret_lag = np.zeros_like(log_close)
            ret_lag[lag:] = log_close[lag:] - log_close[:-lag]
            adf[f"log_return_{lag}"] = ret_lag

        adf["high_low_range"] = log_high - log_low
        adf["open_close_return"] = log_close - log_open

        # Rolling price deviation (12 bars)
        close_series = adf["close"]
        roll_mean_12 = close_series.rolling(12, min_periods=1).mean()
        roll_std_12 = close_series.rolling(12, min_periods=2).std().fillna(1e-4)
        adf["price_dev_12"] = (close_series - roll_mean_12) / np.maximum(roll_std_12, 1e-4)

        # 2. Volatility Features
        # Realized volatility = sqrt(sum of squared returns over backward window)
        ret_sq = adf["log_return_1"] ** 2
        adf["realized_vol_6"] = np.sqrt(ret_sq.rolling(6, min_periods=1).sum())
        adf["realized_vol_12"] = np.sqrt(ret_sq.rolling(12, min_periods=1).sum())
        adf["realized_vol_36"] = np.sqrt(ret_sq.rolling(36, min_periods=1).sum())

        # Parkinson Range Volatility (12 bars)
        hl_sq = (adf["high_low_range"] ** 2)
        park_const = 4.0 * np.log(2.0)
        park_mean = hl_sq.rolling(12, min_periods=1).mean()
        adf["parkinson_vol_12"] = np.sqrt(park_mean / park_const)

        # Volatility ratio
        adf["vol_ratio_6_36"] = adf["realized_vol_6"] / (adf["realized_vol_36"] + 1e-6)

        # 3. Volume and Liquidity Features
        adf["log_volume"] = np.log(vol_p + 1.0)
        vol_series = adf["log_volume"]
        vol_mean_12 = vol_series.rolling(12, min_periods=1).mean()
        vol_std_12 = vol_series.rolling(12, min_periods=2).std().fillna(1e-4)
        adf["volume_zscore_12"] = (vol_series - vol_mean_12) / np.maximum(vol_std_12, 1e-4)
        adf["relative_volume_12"] = (vol_p + 1.0) / (np.exp(vol_mean_12) + 1e-4)
        adf["pv_interaction"] = adf["log_volume"] * np.abs(adf["log_return_1"])

        # 4. Momentum & Reversal
        adf["momentum_6"] = adf["log_return_6"]
        ret_24 = np.zeros_like(log_close)
        ret_24[24:] = log_close[24:] - log_close[:-24]
        adf["momentum_24"] = ret_24

        sign_ret = np.sign(adf["log_return_1"])
        # Proportion of past 12 bars with same sign
        adf["trend_consistency_12"] = (
            (sign_ret == sign_ret.shift(1)).rolling(12, min_periods=1).mean().fillna(0.5)
        )

        out_dfs.append(adf)

    result = pd.concat(out_dfs, ignore_index=True)
    return result
