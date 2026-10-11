"""
Portfolio Accounting Tests
Verifies dollar neutrality, turnover accounting, and friction deductions on hand-calculated toy datasets.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_volatility.portfolio.backtest import DecilePortfolioBacktester
from alpha_volatility.portfolio.costs import TransactionCostModel


def test_hand_calculated_toy_backtest():
    """
    Verifies backtest engine with a 10-asset toy dataset with known returns and signals.
    """
    ts = pd.date_range("2024-01-01 10:00:00", periods=4, freq="5min", tz="UTC")
    records = []

    for t in ts:
        for i in range(10):
            records.append({
                "timestamp": t,
                "asset_id": f"ASSET_{i:02d}",
                # Asset 9 has highest signal (+1.0) and realized return +2%
                # Asset 0 has lowest signal (-1.0) and realized return -2%
                "pred_return_mean": (i - 4.5) * 0.1,
                "target_return": (i - 4.5) * 0.004,
            })

    toy_df = pd.DataFrame(records)

    # 0 transaction costs for pure accounting check
    cost_model = TransactionCostModel(total_one_way_cost_bps=0.0)
    backtester = DecilePortfolioBacktester(
        cost_model=cost_model,
        num_quantiles=10,
        min_universe_size=10,
        rebalance_interval_bars=1,
    )
    res = backtester.run_backtest(toy_df)

    # Long: ASSET_09 (w=+0.5, return = 4.5 * 0.004 = +0.018) -> contribution = +0.009
    # Short: ASSET_00 (w=-0.5, return = -4.5 * 0.004 = -0.018) -> contribution = -0.5 * (-0.018) = +0.009
    # Expected gross return per period = 0.009 + 0.009 = +0.018
    eq = res.equity_curve
    for r in eq["gross_return"]:
        assert np.isclose(r, 0.018, atol=1e-5)

    assert res.is_dollar_neutral
    assert np.isclose(res.monotonicity_rank_corr, 1.0, atol=1e-5)


def test_cost_deduction_impact():
    """Verifies net return is strictly equal to gross return minus transaction costs."""
    ts = pd.date_range("2024-01-01 10:00:00", periods=3, freq="5min", tz="UTC")
    records = []
    for t in ts:
        for i in range(10):
            records.append({
                "timestamp": t,
                "asset_id": f"ASSET_{i:02d}",
                "pred_return_mean": float(i),
                "target_return": 0.001,
            })
    df = pd.DataFrame(records)

    cost_model = TransactionCostModel(total_one_way_cost_bps=10.0)  # 10 bps
    backtester = DecilePortfolioBacktester(cost_model=cost_model, rebalance_interval_bars=1)
    res = backtester.run_backtest(df)

    eq = res.equity_curve
    for _, row in eq.iterrows():
        assert np.isclose(row["net_return"], row["gross_return"] - row["cost"], atol=1e-6)
