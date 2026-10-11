"""
Label Alignment and Information Interval Tracking
Records exact information availability intervals [t, t+h] for purging and embargo.
"""

import pandas as pd

from .returns import compute_future_returns
from .volatility import compute_future_realized_volatility


def attach_label_information_intervals(
    df: pd.DataFrame,
    horizon_bars: int = 3,
) -> pd.DataFrame:
    """
    Computes and attaches explicit start and end timestamps for the label information interval.

    For an observation at index t:
      - label_info_start = timestamp_t
      - label_info_end = timestamp_{t+h}

    Observations at the end where t+h is out-of-bounds receive NaT.

    Args:
        df: DataFrame with 'timestamp' and 'asset_id'.
        horizon_bars: Forecast horizon in bars.

    Returns:
        DataFrame with ['label_info_start', 'label_info_end'] columns.
    """
    out_dfs = []
    for _, asset_df in df.groupby("asset_id", as_index=False):
        adf = asset_df.sort_values("timestamp").copy()
        adf["label_info_start"] = adf["timestamp"]
        adf["label_info_end"] = adf["timestamp"].shift(-horizon_bars)
        out_dfs.append(adf)

    result = pd.concat(out_dfs, ignore_index=True)
    return result


def generate_all_labels(
    df: pd.DataFrame,
    horizon_bars: int = 3,
    drop_unobservable: bool = True,
) -> pd.DataFrame:
    """
    Generates return targets, realized volatility targets, and information intervals.

    Args:
        df: Market bar DataFrame.
        horizon_bars: Forward forecast horizon.
        drop_unobservable: If True, drops trailing rows where target cannot be observed.

    Returns:
        DataFrame with all label columns and alignment intervals.
    """
    labeled = compute_future_returns(df, horizon_bars=horizon_bars)
    labeled = compute_future_realized_volatility(labeled, horizon_bars=horizon_bars)
    labeled = attach_label_information_intervals(labeled, horizon_bars=horizon_bars)

    if drop_unobservable:
        labeled = labeled.dropna(subset=["target_return", "target_realized_vol", "label_info_end"]).reset_index(drop=True)

    labeled = labeled.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)
    return labeled
