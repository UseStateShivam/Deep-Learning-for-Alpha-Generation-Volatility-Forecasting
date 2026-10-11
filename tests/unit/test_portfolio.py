"""
Unit Tests for Portfolio Construction, Dollar Neutrality, and Costs.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_volatility.portfolio.construction import assign_decile_quantiles, construct_dollar_neutral_weights
from alpha_volatility.portfolio.constraints import apply_position_limits
from alpha_volatility.portfolio.costs import TransactionCostModel


def test_dollar_neutral_weights_equality():
    """Verifies that long weights sum to +0.5, short to -0.5, and net equals exactly 0.0."""
    df_ts = pd.DataFrame({
        "asset_id": [f"ASSET_{i:02d}" for i in range(10)],
        "quantile": list(range(1, 11)),
    })
    weights = construct_dollar_neutral_weights(df_ts, num_quantiles=10)
    w_vals = np.array(list(weights.values()))

    long_sum = np.sum(w_vals[w_vals > 0])
    short_sum = np.sum(w_vals[w_vals < 0])
    net_exposure = np.sum(w_vals)
    gross_exposure = np.sum(np.abs(w_vals))

    assert np.isclose(long_sum, 0.5, atol=1e-5)
    assert np.isclose(short_sum, -0.5, atol=1e-5)
    assert np.isclose(net_exposure, 0.0, atol=1e-5)
    assert np.isclose(gross_exposure, 1.0, atol=1e-5)


def test_position_limit_capping():
    """Verifies that position cap rescales weights while preserving neutrality."""
    w = np.array([0.5, -0.5])  # Exceeds 20% cap
    capped = apply_position_limits(w, max_weight_cap=0.20)
    assert np.all(np.abs(capped) <= 0.20001)


def test_transaction_cost_calculation():
    """Verifies linear transaction cost on turnover."""
    cost_model = TransactionCostModel(total_one_way_cost_bps=5.0)  # 5 bps = 0.0005
    w_curr = np.array([0.5, -0.5, 0.0])
    w_target = np.array([0.0, 0.5, -0.5])
    # Turnover traded = |0.0 - 0.5| + |0.5 - (-0.5)| + |-0.5 - 0.0| = 0.5 + 1.0 + 0.5 = 2.0
    # Cost = 2.0 * 0.0005 = 0.0010
    cost = cost_model.calculate_cost(w_target, w_curr)
    assert np.isclose(cost, 0.001, atol=1e-6)
