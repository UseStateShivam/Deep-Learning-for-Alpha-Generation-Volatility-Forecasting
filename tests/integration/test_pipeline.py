"""
Integration Tests
Runs a small, deterministic end-to-end quantitative pipeline from data generation to reporting.
"""

import pytest
from pathlib import Path
from alpha_volatility.data.synthetic import generate_synthetic_market_data
from alpha_volatility.data.validation import validate_bars
from alpha_volatility.features.pipeline import FeaturePipeline
from alpha_volatility.labels.alignment import generate_all_labels
from alpha_volatility.validation.splits import PurgedWalkForwardCV
from alpha_volatility.models.lightgbm_model import LightGBMForecaster
from alpha_volatility.evaluation.forecasting import compute_forecasting_metrics
from alpha_volatility.evaluation.information_coefficient import compute_cross_sectional_ic_series, summarize_ic_metrics
from alpha_volatility.portfolio.backtest import DecilePortfolioBacktester


def test_end_to_end_pipeline():
    """Runs end-to-end fast verification of entire pipeline."""
    # 1. Generate data
    df = generate_synthetic_market_data(num_assets=10, num_sessions=4, bars_per_session=30, seed=42)
    assert len(df) == 10 * 4 * 30

    # 2. Validate data
    val_report = validate_bars(df)
    assert val_report.is_valid

    # 3. Features
    pipeline = FeaturePipeline()
    feat_df = pipeline.generate_features(df)
    assert len(feat_df) == len(df)

    # 4. Labels
    labeled_df = generate_all_labels(feat_df, horizon_bars=3, drop_unobservable=True)
    assert "target_return" in labeled_df.columns
    assert "target_realized_vol" in labeled_df.columns

    # 5. Purged Splitting
    pwf = PurgedWalkForwardCV(n_splits=2, holdout_ratio=0.20, min_train_ratio=0.50)
    dev_df, test_df = pwf.split_holdout(labeled_df)
    folds = pwf.generate_folds(dev_df)
    assert len(folds) == 2

    # 6. Fit Baseline on Fold 1
    scaler = FeaturePipeline.fit_scaler(folds[0].train_df, pipeline.feature_columns)
    s_train = FeaturePipeline.transform_features(folds[0].train_df, scaler, pipeline.feature_columns)
    s_val = FeaturePipeline.transform_features(folds[0].val_df, scaler, pipeline.feature_columns)

    model = LightGBMForecaster(feature_names=pipeline.feature_columns)
    model.fit(s_train, s_val)

    # 7. Predict
    preds = model.predict(s_val)
    val_pred_df = s_val.copy()
    for k, v in preds.items():
        val_pred_df[k] = v

    # 8. Evaluate
    fc_metrics = compute_forecasting_metrics(
        y_true_ret=val_pred_df["target_return"].values,
        y_pred_ret=val_pred_df["pred_return_mean"].values,
        y_true_vol=val_pred_df["target_realized_vol"].values,
        y_pred_vol=val_pred_df["pred_volatility"].values,
    )
    assert "return_rmse" in fc_metrics
    assert "volatility_qlike" in fc_metrics

    ic_series = compute_cross_sectional_ic_series(val_pred_df)
    ic_summary = summarize_ic_metrics(ic_series)
    assert "mean_ic" in ic_summary

    # 9. Backtest
    backtester = DecilePortfolioBacktester()
    res = backtester.run_backtest(val_pred_df)
    assert res.is_dollar_neutral
    assert len(res.equity_curve) > 0
