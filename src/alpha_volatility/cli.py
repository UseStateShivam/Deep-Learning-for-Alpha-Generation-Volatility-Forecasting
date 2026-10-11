"""
Command Line Interface (CLI)
Implements all commands defined in Section 22 for data ingestion, feature generation,
training, evaluation, backtesting, and report generation.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_master_config, load_yaml
from .data.storage import ParquetStorage
from .data.synthetic import generate_synthetic_market_data
from .data.validation import validate_bars
from .evaluation.forecasting import compute_forecasting_metrics
from .evaluation.information_coefficient import (
    compute_cross_sectional_ic_series,
    summarize_ic_metrics,
)
from .evaluation.reports import generate_model_comparison_table, save_metrics_report
from .evaluation.statistical_tests import stationary_block_bootstrap
from .features.pipeline import FeaturePipeline
from .labels.alignment import generate_all_labels
from .logging_config import setup_logger
from .portfolio.backtest import DecilePortfolioBacktester
from .portfolio.costs import TransactionCostModel
from .training.experiment import ExperimentRunner
from .visualization.report_plots import generate_all_report_figures

logger = setup_logger("alpha_volatility.cli")


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validates configuration files."""
    try:
        config = load_master_config(args.config)
        logger.info("Configuration '%s' is valid! Checked all schema parameters.", config.experiment_id)
        return 0
    except Exception as e:
        logger.error("Configuration validation failed: %s", e)
        return 1


def cmd_generate_synthetic_data(args: argparse.Namespace) -> int:
    """Generates deterministic synthetic 5-minute OHLCV bars."""
    try:
        cfg = load_yaml(args.config)
        synth_cfg = cfg.get("synthetic_generator", {})
        universe = cfg.get("universe", {}).get("assets", [f"ASSET_{i:02d}" for i in range(10)])
        storage_dir = cfg.get("storage", {}).get("synthetic_dir", "data/synthetic")

        logger.info("Generating synthetic market bars for %d assets...", len(universe))
        df = generate_synthetic_market_data(
            num_assets=len(universe),
            num_sessions=synth_cfg.get("num_sessions", 60),
            bars_per_session=synth_cfg.get("bars_per_session", 78),
            start_date=synth_cfg.get("start_date", "2024-01-02"),
            seed=synth_cfg.get("seed", 42),
            base_annual_volatility=synth_cfg.get("base_annual_volatility", 0.25),
        )

        storage = ParquetStorage(storage_dir)
        saved_path = storage.save(df, "synthetic_bars.parquet")
        logger.info("Generated %d synthetic bars and saved to %s", len(df), saved_path)
        return 0
    except Exception as e:
        logger.error("Failed to generate synthetic data: %s", e)
        return 1


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Validates market bars in input path."""
    try:
        input_p = Path(args.input)
        if input_p.is_dir():
            files = list(input_p.glob("*.parquet"))
            if not files:
                logger.error("No parquet files found in directory %s", input_p)
                return 1
            file_to_check = files[0]
        else:
            file_to_check = input_p

        df = pd.read_parquet(file_to_check)
        report = validate_bars(df)
        logger.info("Validation report for %s:", file_to_check)
        logger.info("  Valid: %s", report.is_valid)
        logger.info("  Total Bars: %d", report.total_bars)
        logger.info("  Assets: %d (%s)", report.num_assets, ", ".join(report.asset_list[:5]))
        logger.info("  Errors: %d", len(report.errors))
        logger.info("  Warnings: %d", len(report.warnings))
        return 0 if report.is_valid else 1
    except Exception as e:
        logger.error("Failed to validate data: %s", e)
        return 1


def cmd_build_features(args: argparse.Namespace) -> int:
    """Computes point-in-time features and aligns labels."""
    try:
        # Load raw/synthetic data
        data_p = Path("data/synthetic/synthetic_bars.parquet")
        if not data_p.exists():
            data_p = Path("data/raw/market_bars.parquet")
        if not data_p.exists():
            logger.error("No raw or synthetic bars found. Run generate-synthetic-data first.")
            return 1

        df = pd.read_parquet(data_p)
        logger.info("Loaded %d bars for feature engineering...", len(df))

        pipeline = FeaturePipeline()
        feat_df = pipeline.generate_features(df)

        logger.info("Aligning future return and volatility labels...")
        labeled_df = generate_all_labels(feat_df, horizon_bars=3, drop_unobservable=True)

        out_path = Path("data/processed/features_and_labels.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_parquet(out_path, index=False)
        logger.info("Built %d processed feature rows with labels at %s", len(labeled_df), out_path)
        return 0
    except Exception as e:
        logger.error("Failed to build features: %s", e)
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Trains a specified model (tcn, transformer, lightgbm) using purged walk-forward CV."""
    try:
        config = load_master_config(args.config)
        processed_p = Path("data/processed/features_and_labels.parquet")
        if not processed_p.exists():
            logger.info("Processed features not found; building features now...")
            pipeline = FeaturePipeline()
            data_p = Path("data/synthetic/synthetic_bars.parquet")
            if not data_p.exists():
                logger.info("Generating synthetic data first...")
                synth_df = generate_synthetic_market_data(num_assets=10, num_sessions=60)
                ParquetStorage("data/synthetic").save(synth_df, "synthetic_bars.parquet")
                data_p = Path("data/synthetic/synthetic_bars.parquet")
            df = pd.read_parquet(data_p)
            feat_df = pipeline.generate_features(df)
            labeled_df = generate_all_labels(feat_df, horizon_bars=3, drop_unobservable=True)
            processed_p.parent.mkdir(parents=True, exist_ok=True)
            labeled_df.to_parquet(processed_p, index=False)

        data = pd.read_parquet(processed_p)
        runner = ExperimentRunner(config)
        val_preds, test_preds, manifest = runner.run_model_training(args.model.lower(), data)

        # Save predictions
        pred_dir = Path("artifacts/metrics")
        pred_dir.mkdir(parents=True, exist_ok=True)
        val_preds.to_parquet(pred_dir / f"preds_val_{args.model.lower()}.parquet", index=False)
        test_preds.to_parquet(pred_dir / f"preds_test_{args.model.lower()}.parquet", index=False)

        logger.info("Training complete for model %s! Predictions saved to %s", args.model, pred_dir)
        return 0
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluates forecast metrics, cross-sectional IC, and statistical uncertainty."""
    try:
        config = load_master_config(args.config)
        pred_dir = Path("artifacts/metrics")
        models = config.raw_dict.get("active_models", ["lightgbm", "tcn", "transformer"])

        eval_summary = {}

        for m in models:
            pred_file = pred_dir / f"preds_test_{m.lower()}.parquet"
            if not pred_file.exists():
                pred_file = pred_dir / f"preds_val_{m.lower()}.parquet"
            if not pred_file.exists():
                logger.warning("No predictions found for model %s. Skipping.", m)
                continue

            df = pd.read_parquet(pred_file)
            fc_metrics = compute_forecasting_metrics(
                y_true_ret=df["target_return"].values,
                y_pred_ret=df["pred_return_mean"].values,
                y_true_vol=df["target_realized_vol"].values,
                y_pred_vol=df["pred_volatility"].values,
            )

            ic_series = compute_cross_sectional_ic_series(df)
            ic_summary = summarize_ic_metrics(ic_series)

            # Block bootstrap for IC
            if len(ic_series) > 10:
                bs = stationary_block_bootstrap(ic_series["rank_ic"].dropna().values, block_size=78)
            else:
                bs = {"bootstrap_std_error": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

            combined = {**fc_metrics, **ic_summary, "bootstrap_rank_ic": bs}
            eval_summary[m] = combined

        comp_table = generate_model_comparison_table(eval_summary)
        csv_p, json_p = save_metrics_report(comp_table, eval_summary, output_dir=pred_dir)

        logger.info("Evaluation complete. Summary table:\n%s", comp_table.to_string())
        return 0
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return 1


def cmd_backtest(args: argparse.Namespace) -> int:
    """Runs dollar-neutral decile backtest on out-of-sample predictions."""
    try:
        config = load_yaml(args.config)
        pred_dir = Path("artifacts/metrics")

        # Find best available prediction file
        candidate_files = list(pred_dir.glob("preds_test_*.parquet"))
        if not candidate_files:
            candidate_files = list(pred_dir.glob("preds_val_*.parquet"))
        if not candidate_files:
            logger.error("No prediction files found in %s to backtest.", pred_dir)
            return 1

        pred_file = candidate_files[0]
        model_name = pred_file.stem.replace("preds_test_", "").replace("preds_val_", "")
        logger.info("Running decile backtest for model predictions: %s", pred_file.name)

        df = pd.read_parquet(pred_file)
        port_cfg = config.get("portfolio", {})
        cost_cfg = config.get("costs", {})

        cost_model = TransactionCostModel(
            total_one_way_cost_bps=cost_cfg.get("total_one_way_cost_bps", 5.0)
        )
        backtester = DecilePortfolioBacktester(
            cost_model=cost_model,
            num_quantiles=port_cfg.get("num_quantiles", 10),
            min_universe_size=port_cfg.get("min_universe_size", 10),
            rebalance_interval_bars=port_cfg.get("rebalance_interval_bars", 3),
        )

        res = backtester.run_backtest(df)

        # Save backtest results
        res.equity_curve.to_parquet(pred_dir / f"backtest_equity_{model_name}.parquet", index=False)
        res.decile_returns.to_csv(pred_dir / f"decile_returns_{model_name}.csv", index=False)
        with open(pred_dir / f"backtest_metrics_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(res.summary_metrics, f, indent=2)

        logger.info("Backtest complete!")
        logger.info("  Sharpe Ratio (Net): %.2f", res.summary_metrics.get("sharpe_ratio", 0.0))
        logger.info("  Sharpe Ratio (Gross): %.2f", res.summary_metrics.get("sharpe_ratio_gross", 0.0))
        logger.info("  Max Drawdown: %.2f%%", res.summary_metrics.get("max_drawdown", 0.0) * 100.0)
        logger.info("  Monotonicity Score (Rank Corr): %.3f", res.monotonicity_rank_corr)
        logger.info("  Dollar Neutral: %s", res.is_dollar_neutral)
        return 0
    except Exception as e:
        logger.error("Backtest failed: %s", e, exc_info=True)
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generates research figures and prints consolidated research summary."""
    try:
        load_master_config(args.config)
        pred_dir = Path("artifacts/metrics")
        fig_dir = Path("artifacts/figures")

        candidate_preds = list(pred_dir.glob("preds_test_*.parquet")) or list(pred_dir.glob("preds_val_*.parquet"))
        if not candidate_preds:
            logger.error("No predictions found to generate reports.")
            return 1

        pred_df = pd.read_parquet(candidate_preds[0])
        ic_df = compute_cross_sectional_ic_series(pred_df)

        backtester = DecilePortfolioBacktester()
        bt_res = backtester.run_backtest(pred_df)

        feat_cols = FeaturePipeline().feature_columns
        figures = generate_all_report_figures(
            predictions_df=pred_df,
            backtest_result=bt_res,
            ic_df=ic_df,
            feature_cols=feat_cols,
            output_dir=fig_dir,
        )

        logger.info("Successfully generated %d publication-quality research figures in %s", len(figures), fig_dir)
        return 0
    except Exception as e:
        logger.error("Report generation failed: %s", e, exc_info=True)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alpha_volatility",
        description="Deep Learning for Alpha Generation & Volatility Forecasting CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # validate-config
    p_val_cfg = subparsers.add_parser("validate-config", help="Validate YAML experiment configs")
    p_val_cfg.add_argument("--config", default="configs/experiment.yaml", help="Path to config file")
    p_val_cfg.set_defaults(func=cmd_validate_config)

    # generate-synthetic-data
    p_synth = subparsers.add_parser("generate-synthetic-data", help="Generate synthetic market data")
    p_synth.add_argument("--config", default="configs/data.yaml", help="Path to data config")
    p_synth.set_defaults(func=cmd_generate_synthetic_data)

    # validate-data
    p_val_data = subparsers.add_parser("validate-data", help="Validate market bars")
    p_val_data.add_argument("--input", default="data/synthetic/", help="Path to data file or directory")
    p_val_data.set_defaults(func=cmd_validate_data)

    # build-features
    p_feat = subparsers.add_parser("build-features", help="Compute features and align labels")
    p_feat.add_argument("--config", default="configs/features.yaml", help="Path to features config")
    p_feat.set_defaults(func=cmd_build_features)

    # train
    p_train = subparsers.add_parser("train", help="Train forecasting model")
    p_train.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment config")
    p_train.add_argument("--model", required=True, choices=["tcn", "transformer", "lightgbm"], help="Model architecture")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute forecast and IC metrics")
    p_eval.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment config")
    p_eval.set_defaults(func=cmd_evaluate)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run decile portfolio backtest")
    p_bt.add_argument("--config", default="configs/portfolio.yaml", help="Path to portfolio config")
    p_bt.set_defaults(func=cmd_backtest)

    # report
    p_rep = subparsers.add_parser("report", help="Generate research figures and summary report")
    p_rep.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment config")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
