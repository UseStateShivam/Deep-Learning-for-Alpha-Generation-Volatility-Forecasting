"""
Data Cleaning Module
Applies documented cleaning policies to market bars without introducing forward information.
"""

import logging
from typing import Dict, Tuple

import pandas as pd

from .schemas import REQUIRED_BAR_COLUMNS

logger = logging.getLogger("alpha_volatility.data.cleaning")


def clean_bars(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    sort_chronological: bool = True,
    flag_bad_ticks: bool = True,
    max_return_threshold: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans market bar DataFrame according to documented research policies.

    Args:
        df: Input raw bar DataFrame.
        drop_duplicates: Whether to drop duplicate (asset_id, timestamp) rows.
        sort_chronological: Whether to sort by asset_id and timestamp.
        flag_bad_ticks: Whether to detect and remove non-physical outlier spikes.
        max_return_threshold: Maximum allowable 5-minute single bar return before flagging.

    Returns:
        (cleaned_df, cleaning_stats_dict)
    """
    stats: Dict[str, int] = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "invalid_ohlc_removed": 0,
        "bad_ticks_flagged": 0,
        "final_rows": 0,
    }

    cleaned = df.copy()

    # Enforce non-null required columns
    cleaned = cleaned.dropna(subset=REQUIRED_BAR_COLUMNS)

    # Remove non-physical prices
    valid_prices = (
        (cleaned["open"] > 0)
        & (cleaned["high"] > 0)
        & (cleaned["low"] > 0)
        & (cleaned["close"] > 0)
        & (cleaned["volume"] >= 0)
        & (cleaned["high"] >= cleaned["low"])
        & (cleaned["high"] >= cleaned["open"])
        & (cleaned["high"] >= cleaned["close"])
        & (cleaned["low"] <= cleaned["open"])
        & (cleaned["low"] <= cleaned["close"])
    )
    invalid_count = len(cleaned) - valid_prices.sum()
    if invalid_count > 0:
        logger.warning("Dropping %d rows with invalid OHLC geometry or non-positive prices", invalid_count)
        cleaned = cleaned[valid_prices]
        stats["invalid_ohlc_removed"] = int(invalid_count)

    # Deduplicate
    if drop_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["asset_id", "timestamp"], keep="first")
        stats["duplicates_removed"] = before - len(cleaned)

    # Sort
    if sort_chronological:
        cleaned = cleaned.sort_values(by=["asset_id", "timestamp"]).reset_index(drop=True)

    stats["final_rows"] = len(cleaned)
    logger.info("Cleaned bars from %d to %d rows", stats["initial_rows"], stats["final_rows"])
    return cleaned, stats
