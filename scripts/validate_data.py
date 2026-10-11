#!/usr/bin/env python
"""Script to validate data integrity."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from alpha_volatility.data.validation import validate_bars
from alpha_volatility.logging_config import setup_logger

logger = setup_logger("scripts.validate_data")


def main():
    parser = argparse.ArgumentParser(description="Validate market bars against canonical rules.")
    parser.add_argument("--input", default="data/synthetic/synthetic_bars.parquet", help="Path to input parquet")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        logger.error("File does not exist: %s", p)
        sys.exit(1)

    df = pd.read_parquet(p)
    report = validate_bars(df)
    logger.info("Validation outcome: %s", "PASSED" if report.is_valid else "FAILED")
    logger.info("Total bars: %d, Assets: %d", report.total_bars, report.num_assets)
    if report.errors:
        logger.error("Errors found:\n%s", "\n".join(report.errors))
        sys.exit(1)
    if report.warnings:
        logger.warning("Warnings:\n%s", "\n".join(report.warnings))


if __name__ == "__main__":
    main()
