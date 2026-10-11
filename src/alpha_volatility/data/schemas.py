"""
Data Schemas and Contracts
Defines canonical market-data schemas, types, and column contracts.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

REQUIRED_BAR_COLUMNS: List[str] = [
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_BAR_COLUMNS: List[str] = [
    "trade_count",
    "vwap",
    "source",
    "session_id",
]

COLUMN_DTYPES = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "asset_id": "string",
}


@dataclass(frozen=True)
class CanonicalBarSchema:
    """
    Contract for five-minute intraday market bars.

    Convention:
      - 'timestamp': UTC timestamp marking the END (bar close) of the 5-minute interval.
      - Prices (open, high, low, close) are in base currency units (e.g. USD).
      - 'volume' is traded share or contract volume in the interval.
    """
    timestamp_unit: str = "ns"
    timezone: str = "UTC"
    bar_interval: str = "5m"

    @staticmethod
    def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates column existence and standardizes data types.
        """
        missing = [col for col in REQUIRED_BAR_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required canonical columns: {missing}")

        out = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        elif out["timestamp"].dt.tz is None:
            out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
        else:
            out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")

        out["asset_id"] = out["asset_id"].astype(str)
        for num_col in ["open", "high", "low", "close", "volume"]:
            out[num_col] = out[num_col].astype(float)

        return out
