"""
Unit Tests for Evaluation Metrics and Statistical Tests.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_volatility.evaluation.forecasting import compute_forecasting_metrics
from alpha_volatility.evaluation.information_coefficient import compute_cross_sectional_ic_series, summarize_ic_metrics
from alpha_volatility.evaluation.portfolio_metrics import compute_portfolio_performance_metrics, compute_drawdown_series
from alpha_volatility.evaluation.statistical_tests import stationary_block_bootstrap, diebold_mariano_test


def test_forecasting_metrics_mae_rmse():
    """Verifies MAE and RMSE math on synthetic vectors."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    metrics = compute_forecasting_metrics(
        y_true_ret=y_true,
        y_pred_ret=y_pred,
        y_true_vol=np.array([0.1, 0.2, 0.3]),
        y_pred_vol=np.array([0.1, 0.2, 0.3]),
    )
    # Errors: [0.5, 0.0, -0.5] -> MAE = (0.5 + 0.0 + 0.5)/3 = 0.3333...
    assert np.isclose(metrics["return_mae"], 1.0 / 3.0, atol=1e-4)
    # RMSE: sqrt((0.25 + 0.0 + 0.25)/3) = sqrt(0.5/3) = sqrt(1/6) ~= 0.4082
    assert np.isclose(metrics["return_rmse"], np.sqrt(1.0 / 6.0), atol=1e-4)


def test_cross_sectional_ic_known_correlation():
    """Verifies cross-sectional IC when predictions are exactly identical to realized returns."""
    ts = pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [ts[0]] * 5 + [ts[1]] * 5,
        "asset_id": [f"A_{i}" for i in range(5)] * 2,
        "pred_return_mean": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
        "target_return": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
    })
    ic_df = compute_cross_sectional_ic_series(df, min_assets=3)
    assert len(ic_df) == 2
    assert np.allclose(ic_df["ic"], 1.0, atol=1e-5)
    assert np.allclose(ic_df["rank_ic"], 1.0, atol=1e-5)


def test_portfolio_drawdown_calculation():
    """Verifies drawdown calculation on hand-calculated wealth path."""
    wealth = np.array([1.0, 1.2, 1.1, 0.9, 1.0])
    # Peak: 1.0, 1.2, 1.2, 1.2, 1.2
    # Drawdowns: 0, 0, (1.1-1.2)/1.2 = -0.0833, (0.9-1.2)/1.2 = -0.25, -0.1667
    dd, max_dd = compute_drawdown_series(wealth)
    assert np.isclose(max_dd, 0.25, atol=1e-4)


def test_diebold_mariano_superior_model():
    """Verifies DM test detects significant loss difference between superior and inferior models."""
    n = 200
    loss_inferior = np.ones(n) * 1.0
    loss_superior = np.ones(n) * 0.1
    stat, p_val = diebold_mariano_test(loss_inferior, loss_superior)
    assert stat > 5.0
    assert p_val < 0.01
