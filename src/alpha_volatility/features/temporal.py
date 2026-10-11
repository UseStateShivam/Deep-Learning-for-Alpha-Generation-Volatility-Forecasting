"""
Temporal Features Module
Extracts cyclical and point-in-time time indicators without using future session information.
"""

import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame, bars_per_session: int = 78) -> pd.DataFrame:
    """
    Computes time-of-day and calendar features from bar timestamps.

    Args:
        df: DataFrame with 'timestamp' column (UTC).
        bars_per_session: Expected number of 5m bars in a regular session (default 78 for 6.5h).

    Returns:
        DataFrame with temporal features appended.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)

    # Intraday minutes from market open (assumes standard 14:30 UTC = 09:30 EST)
    # Generic extraction using minute-of-day:
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    # Map into session bar index assuming ~390 minutes (6.5 hours) trading window
    # Normalize cyclical intraday time
    bar_of_day = (minute_of_day % 390) / 5.0
    progress = bar_of_day / float(bars_per_session)
    progress = np.clip(progress, 0.0, 1.0)

    out["sin_time_of_day"] = np.sin(2.0 * np.pi * progress)
    out["cos_time_of_day"] = np.cos(2.0 * np.pi * progress)
    out["session_progress"] = progress

    # Day of week (0 = Monday, 4 = Friday)
    dow = ts.dt.dayofweek
    out["day_of_week_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    out["day_of_week_cos"] = np.cos(2.0 * np.pi * dow / 5.0)

    return out
