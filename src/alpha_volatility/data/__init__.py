"""
Data module for canonical OHLCV ingestion, validation, synthetic generation, and storage.
"""

from .cleaning import clean_bars
from .providers import BaseMarketDataProvider, SyntheticDataProvider, YahooFinanceProvider
from .schemas import REQUIRED_BAR_COLUMNS, CanonicalBarSchema
from .storage import ParquetStorage
from .synthetic import generate_synthetic_market_data
from .validation import DataValidationError, validate_bars

__all__ = [
    "CanonicalBarSchema",
    "REQUIRED_BAR_COLUMNS",
    "validate_bars",
    "DataValidationError",
    "clean_bars",
    "ParquetStorage",
    "BaseMarketDataProvider",
    "YahooFinanceProvider",
    "SyntheticDataProvider",
    "generate_synthetic_market_data",
]
