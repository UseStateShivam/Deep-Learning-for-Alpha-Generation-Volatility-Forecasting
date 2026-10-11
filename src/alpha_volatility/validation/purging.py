"""
Purging Module
Removes training observations whose label information intervals overlap with validation evaluation periods.
"""

from typing import Tuple

import pandas as pd


def purge_overlapping_samples(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    info_start_col: str = "label_info_start",
    info_end_col: str = "label_info_end",
) -> Tuple[pd.DataFrame, int]:
    """
    Purges any sample in train_df whose label information interval overlaps
    with ANY validation interval in val_df.

    Interval overlap condition between [T_s, T_e] (train sample) and [V_s, V_e] (validation window):
      Overlap occurs if:
        T_e >= V_s  AND  T_s <= V_e

    Args:
        train_df: Candidate training samples with info start/end timestamps.
        val_df: Validation samples with info start/end timestamps.
        info_start_col: Start column of label information interval.
        info_end_col: End column of label information interval.

    Returns:
        (purged_train_df, num_purged_samples)
    """
    if train_df.empty or val_df.empty:
        return train_df.copy(), 0

    val_start = val_df[info_start_col].min()
    val_end = val_df[info_end_col].max()

    train_start = train_df[info_start_col]
    train_end = train_df[info_end_col]

    # Two closed intervals [T_s, T_e] and [V_s, V_e] overlap iff:
    # max(T_s, V_s) <= min(T_e, V_e)
    # which is equivalent to: (train_end >= val_start) & (train_start <= val_end)
    overlaps = (train_end >= val_start) & (train_start <= val_end)
    num_purged = int(overlaps.sum())

    purged_train = train_df[~overlaps].copy()
    return purged_train, num_purged
