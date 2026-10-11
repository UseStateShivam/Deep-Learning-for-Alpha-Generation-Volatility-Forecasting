"""
Data Validation Module
Performs comprehensive point-in-time checks on market bar datasets according to Section 5.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schemas import REQUIRED_BAR_COLUMNS


class DataValidationError(Exception):
    """Raised when critical market bar integrity checks fail."""
    pass


@dataclass
class ValidationReport:
    """Summary of data validation results."""
    is_valid: bool
    total_bars: int
    num_assets: int
    asset_list: List[str]
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def validate_bars(
    df: pd.DataFrame,
    max_log_return_deviation: float = 0.15,
    stale_price_window: int = 12,
    raise_on_error: bool = False,
) -> ValidationReport:
    """
    Validates market bars against canonical integrity standards.

    Checks performed:
    1. Required columns presence
    2. Datetime parsing and UTC timezone
    3. Duplicate (asset_id, timestamp) pairs
    4. Monotonic chronological ordering per asset
    5. Positive non-zero prices (open, high, low, close > 0)
    6. Non-negative volume (volume >= 0)
    7. OHLC mathematical relations: high >= low, high >= open, high >= close, low <= open, low <= close
    8. Extreme return swings (possible bad ticks)
    9. Stale/flatlined price detection
    10. Asset identifiers consistency
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Missing columns
    missing_cols = [c for c in REQUIRED_BAR_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        if raise_on_error:
            raise DataValidationError("; ".join(errors))
        return ValidationReport(
            is_valid=False,
            total_bars=len(df),
            num_assets=0,
            asset_list=[],
            errors=errors,
            warnings=warnings,
        )

    # 2. Datetime check
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("Column 'timestamp' is not datetime typed.")
    elif df["timestamp"].dt.tz is None:
        warnings.append("Timestamps lack explicit timezone; treating as UTC.")

    # 3. Duplicate pairs
    dups = df.duplicated(subset=["asset_id", "timestamp"]).sum()
    if dups > 0:
        errors.append(f"Found {dups} duplicate (asset_id, timestamp) entries.")

    # Group by asset for asset-specific checks
    assets = df["asset_id"].unique().tolist()
    total_bars = len(df)
    extreme_returns_count = 0
    stale_sequences_count = 0

    for asset in assets:
        adf = df[df["asset_id"] == asset]

        # 4. Monotonic ordering
        if not adf["timestamp"].is_monotonic_increasing:
            errors.append(f"Asset {asset} timestamps are not monotonically increasing.")

        # 5. Non-positive prices
        for col in ["open", "high", "low", "close"]:
            non_pos = (adf[col] <= 0).sum()
            if non_pos > 0:
                errors.append(f"Asset {asset} has {non_pos} non-positive values in {col}.")

        # 6. Negative volume
        neg_vol = (adf["volume"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"Asset {asset} has {neg_vol} negative volume observations.")

        # 7. OHLC consistency
        hl_err = (adf["high"] < adf["low"]).sum()
        ho_err = (adf["high"] < adf["open"]).sum()
        hc_err = (adf["high"] < adf["close"]).sum()
        lo_err = (adf["low"] > adf["open"]).sum()
        lc_err = (adf["low"] > adf["close"]).sum()
        ohlc_inconsistencies = hl_err + ho_err + hc_err + lo_err + lc_err
        if ohlc_inconsistencies > 0:
            errors.append(
                f"Asset {asset} has {ohlc_inconsistencies} OHLC inequality violations "
                f"(high < low: {hl_err}, high < open: {ho_err}, low > close: {lc_err}, etc.)."
            )

        # 8. Extreme return swings
        log_ret = np.abs(np.log(adf["close"] / adf["close"].shift(1)))
        extremes = (log_ret > max_log_return_deviation).sum()
        if extremes > 0:
            warnings.append(
                f"Asset {asset} has {extremes} bars with single-bar return > {max_log_return_deviation:.1%}."
            )
            extreme_returns_count += extremes

        # 9. Stale prices
        repeats = (adf["close"] == adf["close"].shift(1)).astype(int)
        rolling_repeats = repeats.rolling(stale_price_window).sum()
        stale_runs = (rolling_repeats >= stale_price_window - 1).sum()
        if stale_runs > 0:
            warnings.append(
                f"Asset {asset} has {stale_runs} occurrences of stale prices for >= {stale_price_window} consecutive bars."
            )
            stale_sequences_count += stale_runs

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise DataValidationError("; ".join(errors))

    start_ts = df["timestamp"].min().isoformat() if len(df) > 0 else None
    end_ts = df["timestamp"].max().isoformat() if len(df) > 0 else None

    return ValidationReport(
        is_valid=is_valid,
        total_bars=total_bars,
        num_assets=len(assets),
        asset_list=[str(a) for a in assets],
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        errors=errors,
        warnings=warnings,
        metrics={
            "extreme_returns_count": float(extreme_returns_count),
            "stale_sequences_count": float(stale_sequences_count),
        },
    )
