#!/usr/bin/env python
"""Script to download real market data or generate synthetic bars."""

import argparse
import sys
from alpha_volatility.data.ingestion import ingest_market_data
from alpha_volatility.config import load_yaml
from alpha_volatility.logging_config import setup_logger

logger = setup_logger("scripts.download_data")


def main():
    parser = argparse.ArgumentParser(description="Download or generate market bars.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df, report, saved_path = ingest_market_data(cfg)
    logger.info("Ingestion completed. Saved to %s (Valid: %s)", saved_path, report.is_valid)


if __name__ == "__main__":
    main()
