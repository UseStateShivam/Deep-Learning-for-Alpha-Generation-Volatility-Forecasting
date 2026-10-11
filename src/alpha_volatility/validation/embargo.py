"""
Embargo Module
Enforces post-validation embargo periods to eliminate autoregressive serial contamination.
"""

from typing import Tuple

import pandas as pd


def apply_embargo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    embargo_bars: int = 12,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, int]:
    """
    Applies an embargo buffer after the validation period.
    Any training observation occurring strictly within [val_end, val_end + embargo_delta]
    is excluded.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        embargo_bars: Number of 5m bars to embargo (e.g. 12 bars = 60 minutes).
        timestamp_col: Timestamp column.

    Returns:
        (embargoed_train_df, num_embargoed_samples)
    """
    if train_df.empty or val_df.empty or embargo_bars <= 0:
        return train_df.copy(), 0

    val_end = val_df[timestamp_col].max()
    # Assume 5-minute bars: embargo delta = embargo_bars * 5 minutes
    embargo_delta = pd.Timedelta(minutes=5 * embargo_bars)
    embargo_cutoff = val_end + embargo_delta

    # Training samples that occur between val_end and embargo_cutoff are embargoed
    train_ts = train_df[timestamp_col]
    embargo_mask = (train_ts > val_end) & (train_ts <= embargo_cutoff)
    num_embargoed = int(embargo_mask.sum())

    embargoed_train = train_df[~embargo_mask].copy()
    return embargoed_train, num_embargoed
