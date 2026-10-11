#!/usr/bin/env python
"""Script to run full walk-forward model training and evaluation."""

import argparse
from pathlib import Path
import pandas as pd
from alpha_volatility.config import load_master_config
from alpha_volatility.training.experiment import ExperimentRunner
from alpha_volatility.logging_config import setup_logger

logger = setup_logger("scripts.run_experiment")


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward model experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--model", default="tcn", choices=["tcn", "transformer", "lightgbm"], help="Model to train")
    args = parser.parse_args()

    cfg = load_master_config(args.config)
    proc_file = Path("data/processed/features_and_labels.parquet")
    if not proc_file.exists():
        logger.error("Features not built. Run 'alpha-volatility build-features' first.")
        return

    df = pd.read_parquet(proc_file)
    runner = ExperimentRunner(cfg)
    val_preds, test_preds, manifest = runner.run_model_training(args.model, df)

    pred_dir = Path("artifacts/metrics")
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_preds.to_parquet(pred_dir / f"preds_val_{args.model}.parquet", index=False)
    test_preds.to_parquet(pred_dir / f"preds_test_{args.model}.parquet", index=False)
    logger.info("Experiment completed successfully for %s!", args.model)


if __name__ == "__main__":
    main()
