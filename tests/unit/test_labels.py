"""
Unit Tests for Target and Label Engineering.
"""

import numpy as np
import pandas as pd

from alpha_volatility.labels.alignment import attach_label_information_intervals
from alpha_volatility.labels.returns import compute_future_returns
from alpha_volatility.labels.volatility import compute_future_realized_volatility


def test_forward_return_calculation(sample_market_bars):
    """Verifies that forward log return r(t, h) matches ln(P(t+h) / P(t))."""
    h = 3
    labeled = compute_future_returns(sample_market_bars, horizon_bars=h)
    assert "target_return" in labeled.columns

    # Test asset 0
    asset_sub = labeled[labeled["asset_id"] == "ASSET_00"].sort_values("timestamp")
    t0_close = asset_sub["close"].iloc[0]
    t3_close = asset_sub["close"].iloc[3]
    expected_ret = np.log(t3_close / t0_close)
    actual_ret = asset_sub["target_return"].iloc[0]
    assert np.isclose(expected_ret, actual_ret, atol=1e-6)

    # Trailing h rows should be NaN
    assert np.isnan(asset_sub["target_return"].iloc[-1])
    assert np.isnan(asset_sub["target_return"].iloc[-h])


def test_realized_volatility_calculation(sample_market_bars):
    """Verifies realized volatility formula sqrt(sum(r_{t+i}^2)) over forward horizon."""
    h = 3
    labeled = compute_future_realized_volatility(sample_market_bars, horizon_bars=h)
    assert "target_realized_vol" in labeled.columns
    assert "target_log_realized_vol" in labeled.columns

    asset_sub = labeled[labeled["asset_id"] == "ASSET_00"].sort_values("timestamp")
    p0 = asset_sub["close"].iloc[0]
    p1 = asset_sub["close"].iloc[1]
    p2 = asset_sub["close"].iloc[2]
    p3 = asset_sub["close"].iloc[3]

    r1 = np.log(p1 / p0)
    r2 = np.log(p2 / p1)
    r3 = np.log(p3 / p2)
    expected_rv = np.sqrt(r1**2 + r2**2 + r3**2)
    actual_rv = asset_sub["target_realized_vol"].iloc[0]
    assert np.isclose(expected_rv, actual_rv, atol=1e-5)


def test_label_information_intervals(sample_market_bars):
    """Verifies label information start is t and end is t+h."""
    h = 3
    aligned = attach_label_information_intervals(sample_market_bars, horizon_bars=h)
    assert "label_info_start" in aligned.columns
    assert "label_info_end" in aligned.columns

    asset_sub = aligned[aligned["asset_id"] == "ASSET_00"].sort_values("timestamp")
    assert asset_sub["label_info_start"].iloc[0] == asset_sub["timestamp"].iloc[0]
    assert asset_sub["label_info_end"].iloc[0] == asset_sub["timestamp"].iloc[h]
    assert pd.isna(asset_sub["label_info_end"].iloc[-1])
