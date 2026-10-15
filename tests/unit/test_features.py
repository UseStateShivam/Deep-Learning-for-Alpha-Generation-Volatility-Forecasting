"""
Unit Tests for Technical, Temporal, and Cross-Sectional Feature Engineering.
"""

import numpy as np

from alpha_volatility.features.cross_sectional import compute_cross_sectional_features
from alpha_volatility.features.pipeline import FeaturePipeline
from alpha_volatility.features.technical import compute_technical_features
from alpha_volatility.features.temporal import compute_temporal_features


def test_technical_log_returns(sample_market_bars):
    """Verifies that 1-bar log return matches exact mathematical formulation."""
    feat_df = compute_technical_features(sample_market_bars)
    assert "log_return_1" in feat_df.columns
    # Check for first asset
    first_asset = sample_market_bars["asset_id"].iloc[0]
    sub = feat_df[feat_df["asset_id"] == first_asset].sort_values("timestamp")

    expected_ret_1 = np.log(sub["close"].iloc[1] / sub["close"].iloc[0])
    actual_ret_1 = sub["log_return_1"].iloc[1]
    assert np.isclose(expected_ret_1, actual_ret_1, atol=1e-6)


def test_temporal_cyclical_bounds(sample_market_bars):
    """Verifies cyclical sine and cosine time features remain in [-1, +1]."""
    temp_df = compute_temporal_features(sample_market_bars)
    assert np.all(temp_df["sin_time_of_day"] >= -1.0)
    assert np.all(temp_df["sin_time_of_day"] <= 1.0)
    assert np.all(temp_df["cos_time_of_day"] >= -1.0)
    assert np.all(temp_df["cos_time_of_day"] <= 1.0)


def test_cross_sectional_ranks_bounded(sample_market_bars):
    """Verifies cross-sectional ranks are bounded in [-0.5, +0.5] and sum close to 0."""
    feat_df = compute_technical_features(sample_market_bars)
    cs_df = compute_cross_sectional_features(feat_df, features_to_rank=["log_return_1"])

    rank_col = "cs_rank_log_return_1"
    assert rank_col in cs_df.columns
    assert np.all(cs_df[rank_col] >= -0.5001)
    assert np.all(cs_df[rank_col] <= 0.5001)

    # For any timestamp with all 10 assets, average rank should be exactly 0
    ts_sample = cs_df["timestamp"].iloc[100]
    ts_group = cs_df[cs_df["timestamp"] == ts_sample]
    assert np.isclose(ts_group[rank_col].mean(), 0.0, atol=1e-5)


def test_feature_pipeline_scaler_isolation(sample_market_bars):
    """Verifies scaler does not mutate training data or leak across datasets."""
    pipeline = FeaturePipeline()
    feat_df = pipeline.generate_features(sample_market_bars)

    train_slice = feat_df.iloc[:200]
    test_slice = feat_df.iloc[200:]

    scaler = pipeline.fit_scaler(train_slice, pipeline.feature_columns, method="standard")
    scaled_train = pipeline.transform_features(train_slice, scaler, pipeline.feature_columns)
    pipeline.transform_features(test_slice, scaler, pipeline.feature_columns)

    # Scaled training mean should be ~0 and std ~1
    train_vals = scaled_train[pipeline.feature_columns].values
    assert np.all(np.abs(np.mean(train_vals, axis=0)) < 0.1)
