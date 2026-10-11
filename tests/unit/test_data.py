"""
Unit Tests for Data Ingestion, Validation, Cleaning, and Storage.
"""

import pandas as pd

from alpha_volatility.data.cleaning import clean_bars
from alpha_volatility.data.providers import BaseMarketDataProvider, SyntheticDataProvider
from alpha_volatility.data.schemas import REQUIRED_BAR_COLUMNS, CanonicalBarSchema
from alpha_volatility.data.storage import ParquetStorage
from alpha_volatility.data.synthetic import generate_synthetic_market_data
from alpha_volatility.data.validation import validate_bars


def test_synthetic_data_generation_deterministic():
    """Verifies that synthetic data generation is fully deterministic across identical seeds."""
    df1 = generate_synthetic_market_data(num_assets=3, num_sessions=2, seed=42)
    df2 = generate_synthetic_market_data(num_assets=3, num_sessions=2, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_canonical_schema_enforcement(sample_market_bars):
    """Verifies all required canonical columns and types are present."""
    df = CanonicalBarSchema.enforce_schema(sample_market_bars)
    for col in REQUIRED_BAR_COLUMNS:
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].dt.tz is not None


def test_validation_detects_ohlc_inconsistency():
    """Verifies validation detects high < low or high < close violations."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
        "asset_id": ["A"] * 5,
        "open": [10.0, 10.0, 10.0, 10.0, 10.0],
        "high": [9.0, 10.0, 10.0, 10.0, 10.0],  # high < open / low!
        "low": [10.0, 10.0, 10.0, 10.0, 10.0],
        "close": [10.0, 10.0, 10.0, 10.0, 10.0],
        "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    report = validate_bars(df)
    assert not report.is_valid
    assert any("OHLC" in err for err in report.errors)


def test_validation_detects_non_positive_price():
    """Verifies validation flags zero or negative prices."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
        "asset_id": ["A"] * 3,
        "open": [10.0, -1.0, 10.0],
        "high": [11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0],
        "close": [10.0, 10.0, 10.0],
        "volume": [100.0, 100.0, 100.0],
    })
    report = validate_bars(df)
    assert not report.is_valid
    assert any("non-positive" in err for err in report.errors)


def test_clean_bars_deduplication():
    """Verifies duplicate timestamps for same asset are removed."""
    ts = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [ts[0], ts[0], ts[1]],
        "asset_id": ["A", "A", "A"],
        "open": [10.0, 10.0, 10.0],
        "high": [11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0],
        "close": [10.0, 10.0, 10.0],
        "volume": [100.0, 100.0, 100.0],
    })
    cleaned, stats = clean_bars(df)
    assert len(cleaned) == 2
    assert stats["duplicates_removed"] == 1


def test_parquet_storage_roundtrip(tmp_path, sample_market_bars):
    """Verifies Parquet saving, hashing, and loading."""
    storage = ParquetStorage(tmp_path)
    saved_p = storage.save(sample_market_bars, "test_bars.parquet")
    assert saved_p.exists()

    loaded_df = storage.load("test_bars.parquet")
    assert len(loaded_df) == len(sample_market_bars)
    assert storage.compute_dataframe_hash(sample_market_bars) == storage.compute_dataframe_hash(loaded_df)


def test_synthetic_data_provider_polymorphic_call():
    """Verifies that SyntheticDataProvider can be invoked polymorphically via BaseMarketDataProvider with num_sessions."""
    provider: BaseMarketDataProvider = SyntheticDataProvider(seed=42)
    df = provider.fetch_bars(symbols=["ASSET_00"], start_date="2024-01-02", num_sessions=5)
    assert not df.empty
    assert "asset_id" in df.columns
    assert (df["asset_id"] == "ASSET_00").all()

