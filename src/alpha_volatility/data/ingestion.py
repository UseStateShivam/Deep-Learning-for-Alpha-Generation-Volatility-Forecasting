"""
Data Ingestion Pipeline
Coordinates data retrieval, validation, cleaning, and storage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .cleaning import clean_bars
from .providers import BaseMarketDataProvider, SyntheticDataProvider, YahooFinanceProvider
from .storage import ParquetStorage
from .validation import ValidationReport, validate_bars

logger = logging.getLogger("alpha_volatility.data.ingestion")


def ingest_market_data(
    data_config: Dict[str, Any],
    output_filename: str = "market_bars.parquet",
) -> Tuple[pd.DataFrame, ValidationReport, Path]:
    """
    Executes end-to-end data ingestion pipeline:
    1. Instantiates the configured provider.
    2. Retrieves bars.
    3. Validates raw bars.
    4. Applies cleaning policies.
    5. Saves to Parquet storage with metadata sidecar.

    Args:
        data_config: Dictionary loaded from configs/data.yaml.
        output_filename: Target Parquet filename.

    Returns:
        (cleaned_df, validation_report, saved_file_path)
    """
    provider_name = data_config.get("provider", {}).get("name", "synthetic")
    universe = data_config.get("universe", {}).get("assets", ["ASSET_00"])
    storage_dir = data_config.get("storage", {}).get(
        "synthetic_dir" if provider_name == "synthetic" else "processed_dir", "data/processed"
    )

    logger.info("Starting ingestion with provider '%s' for universe of %d assets", provider_name, len(universe))

    if provider_name == "synthetic":
        synth_cfg = data_config.get("synthetic_generator", {})
        provider: BaseMarketDataProvider = SyntheticDataProvider(
            seed=synth_cfg.get("seed", 42),
            base_annual_volatility=synth_cfg.get("base_annual_volatility", 0.25),
        )
        raw_df = provider.fetch_bars(
            symbols=universe,
            start_date=synth_cfg.get("start_date", "2024-01-02"),
            num_sessions=synth_cfg.get("num_sessions", 60),
        )
    elif provider_name == "yfinance":
        provider = YahooFinanceProvider()
        raw_df = provider.fetch_bars(
            symbols=universe,
            start_date="2024-01-01",
        )
    else:
        raise ValueError(f"Unknown market data provider: {provider_name}")

    # Validate raw data
    val_report = validate_bars(raw_df)
    logger.info(
        "Validation completed. Is valid: %s, errors: %d, warnings: %d",
        val_report.is_valid,
        len(val_report.errors),
        len(val_report.warnings),
    )

    # Clean data
    cleaned_df, clean_stats = clean_bars(raw_df)

    # Persist
    storage = ParquetStorage(storage_dir)
    meta = {
        "provider": provider.get_metadata(),
        "cleaning_stats": clean_stats,
        "validation_metrics": val_report.metrics,
    }
    saved_path = storage.save(cleaned_df, output_filename, metadata=meta)
    logger.info("Successfully ingested and saved %d bars to %s", len(cleaned_df), saved_path)

    return cleaned_df, val_report, saved_path
