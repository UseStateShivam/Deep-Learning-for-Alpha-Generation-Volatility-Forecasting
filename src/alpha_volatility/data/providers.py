"""
Market Data Providers
Abstract interface and implementations for synthetic and real market data ingestion.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from .schemas import CanonicalBarSchema
from .synthetic import generate_synthetic_market_data


class BaseMarketDataProvider(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def fetch_bars(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "5m",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetches canonical OHLCV bars for requested symbols."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns provider metadata including licensing and timestamp conventions."""
        pass


class SyntheticDataProvider(BaseMarketDataProvider):
    """Deterministic synthetic intraday provider for unit testing and CI reproduction."""

    def __init__(self, seed: int = 42, base_annual_volatility: float = 0.25):
        self.seed = seed
        self.base_annual_volatility = base_annual_volatility

    def fetch_bars(
        self,
        symbols: List[str],
        start_date: str = "2024-01-02",
        end_date: Optional[str] = None,
        interval: str = "5m",
        num_sessions: int = 60,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df = generate_synthetic_market_data(
            num_assets=len(symbols),
            num_sessions=num_sessions,
            start_date=start_date,
            seed=self.seed,
            base_annual_volatility=self.base_annual_volatility,
        )
        # Rename asset IDs if symbols specified
        if symbols:
            unique_assets = df["asset_id"].unique()
            mapping = {old: new for old, new in zip(unique_assets, symbols, strict=False)}
            df["asset_id"] = df["asset_id"].map(mapping).fillna(df["asset_id"])

        return CanonicalBarSchema.enforce_schema(df)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider_name": "SyntheticDataProvider",
            "is_synthetic": True,
            "seed": self.seed,
            "timezone": "UTC",
            "licensing": "Generated purely for local deterministic testing and verification.",
        }


class YahooFinanceProvider(BaseMarketDataProvider):
    """
    Real market data provider using Yahoo Finance public API.
    Note: Yahoo Finance public 5-minute intraday data is typically limited to the last 60 calendar days.
    """

    def __init__(self):
        pass

    def fetch_bars(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "5m",
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for YahooFinanceProvider. Install with `pip install yfinance`."
            ) from None

        records: List[pd.DataFrame] = []
        for sym in symbols:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start_date, end=end_date, interval=interval)
            if hist.empty:
                continue

            hist = hist.reset_index()
            # Normalize column names
            time_col = "Datetime" if "Datetime" in hist.columns else "Date"
            hist = hist.rename(
                columns={
                    time_col: "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            hist["asset_id"] = sym
            records.append(hist[["timestamp", "asset_id", "open", "high", "low", "close", "volume"]])

        if not records:
            raise ValueError(f"No bars returned for symbols: {symbols}")

        combined = pd.concat(records, ignore_index=True)
        return CanonicalBarSchema.enforce_schema(combined)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider_name": "YahooFinanceProvider",
            "is_synthetic": False,
            "timezone": "UTC",
            "licensing": "Public market data via Yahoo Finance. For research and personal educational use only.",
        }
