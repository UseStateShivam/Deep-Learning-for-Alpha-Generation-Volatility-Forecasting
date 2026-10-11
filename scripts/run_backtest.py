#!/usr/bin/env python
"""Script to run decile portfolio simulation and backtest."""

import argparse
from pathlib import Path
import pandas as pd
from alpha_volatility.config import load_yaml
from alpha_volatility.portfolio.backtest import DecilePortfolioBacktester
from alpha_volatility.portfolio.costs import TransactionCostModel
from alpha_volatility.logging_config import setup_logger

logger = setup_logger("scripts.run_backtest")


def main():
    parser = argparse.ArgumentParser(description="Run dollar-neutral decile backtest.")
    parser.add_argument("--config", default="configs/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument("--predictions", default="artifacts/metrics/preds_test_tcn.parquet", help="Path to predictions parquet")
    args = parser.parse_args()

    pred_p = Path(args.predictions)
    if not pred_p.exists():
        cand = list(Path("artifacts/metrics").glob("preds_*.parquet"))
        if cand:
            pred_p = cand[0]
        else:
            logger.error("No predictions found at %s", args.predictions)
            return

    df = pd.read_parquet(pred_p)
    cfg = load_yaml(args.config)
    cost_model = TransactionCostModel(
        total_one_way_cost_bps=cfg.get("costs", {}).get("total_one_way_cost_bps", 5.0)
    )
    backtester = DecilePortfolioBacktester(cost_model=cost_model)
    res = backtester.run_backtest(df)

    logger.info("Backtest results:")
    logger.info("  Cumulative Return (Net): %.2f%%", res.summary_metrics.get("cumulative_return", 0.0) * 100)
    logger.info("  Sharpe Ratio (Net): %.2f", res.summary_metrics.get("sharpe_ratio", 0.0))
    logger.info("  Sharpe Ratio (Gross): %.2f", res.summary_metrics.get("sharpe_ratio_gross", 0.0))
    logger.info("  Max Drawdown: %.2f%%", res.summary_metrics.get("max_drawdown", 0.0) * 100)
    logger.info("  Decile Monotonicity (Spearman Rank Corr): %.3f", res.monotonicity_rank_corr)


if __name__ == "__main__":
    main()
