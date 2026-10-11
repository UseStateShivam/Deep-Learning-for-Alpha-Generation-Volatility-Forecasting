#!/usr/bin/env python
"""Script to generate research report and plots."""

import argparse
from pathlib import Path
import pandas as pd
from alpha_volatility.config import load_master_config
from alpha_volatility.portfolio.backtest import DecilePortfolioBacktester
from alpha_volatility.evaluation.information_coefficient import compute_cross_sectional_ic_series
from alpha_volatility.visualization.report_plots import generate_all_report_figures
from alpha_volatility.features.pipeline import FeaturePipeline
from alpha_volatility.logging_config import setup_logger

logger = setup_logger("scripts.generate_report")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready research report plots.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment config")
    args = parser.parse_args()

    cfg = load_master_config(args.config)
    pred_dir = Path("artifacts/metrics")
    preds = list(pred_dir.glob("preds_*.parquet"))
    if not preds:
        logger.error("No predictions found in %s", pred_dir)
        return

    df = pd.read_parquet(preds[0])
    ic_df = compute_cross_sectional_ic_series(df)
    bt = DecilePortfolioBacktester().run_backtest(df)
    feat_cols = FeaturePipeline().feature_columns

    figs = generate_all_report_figures(
        predictions_df=df,
        backtest_result=bt,
        ic_df=ic_df,
        feature_cols=feat_cols,
        output_dir="artifacts/figures",
    )
    logger.info("Generated %d research figures in artifacts/figures/", len(figs))


if __name__ == "__main__":
    main()
