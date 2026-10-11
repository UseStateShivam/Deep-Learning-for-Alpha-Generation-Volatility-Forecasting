"""
Synthetic Market Data Generator
Generates realistic, deterministic intraday 5-minute OHLCV market bars
with volatility clustering, regime switching, U-shaped intraday volume curves,
and controlled lead-lag alpha relationships for testing and reproducibility.
"""

from typing import List

import numpy as np
import pandas as pd


def generate_synthetic_market_data(
    num_assets: int = 10,
    num_sessions: int = 60,
    bars_per_session: int = 78,
    start_date: str = "2024-01-02",
    seed: int = 42,
    base_annual_volatility: float = 0.25,
    regime_switch_prob: float = 0.05,
    autocorrelation_coeff: float = 0.04,
    signal_strength: float = 0.08,
) -> pd.DataFrame:
    """
    Generates deterministic multi-asset 5-minute OHLCV market bars.

    Features simulated:
    - 6.5-hour trading sessions (78 bars per session at 5m frequency).
    - Geometric Brownian motion with regime-dependent stochastic volatility (low vs high vol).
    - Intraday U-shaped volume distribution (high at open and close, low at midday).
    - Controlled weak momentum/mean-reversion component to verify alpha signals.
    - Canonical OHLC geometry strictly maintained: High >= max(Open, Close) and Low <= min(Open, Close).

    Args:
        num_assets: Number of distinct assets in the universe.
        num_sessions: Number of consecutive business days to simulate.
        bars_per_session: Number of 5m bars per trading day (default 78 = 6.5 hours).
        start_date: Starting calendar date.
        seed: Deterministic random seed.
        base_annual_volatility: Base volatility annualized.
        regime_switch_prob: Probability of Markov transition between volatility regimes.
        autocorrelation_coeff: Microstructure AR(1) parameter.
        signal_strength: Latent factor strength for realistic alpha predictability.

    Returns:
        pd.DataFrame containing canonical bar columns.
    """
    rng = np.random.default_rng(seed)
    asset_ids = [f"ASSET_{i:02d}" for i in range(num_assets)]

    # Generate trading timestamps: e.g. 09:30 to 16:00 EST equivalent in UTC: 14:30 to 21:00 UTC
    session_dates = pd.bdate_range(start=start_date, periods=num_sessions)
    timestamps: List[pd.Timestamp] = []

    for s_date in session_dates:
        # Market open 14:30 UTC, bar close begins at 14:35 UTC (after first 5 mins)
        start_time = s_date.tz_localize("UTC") + pd.Timedelta(hours=14, minutes=35)
        for b in range(bars_per_session):
            timestamps.append(start_time + pd.Timedelta(minutes=5 * b))

    total_bars = len(timestamps)

    # U-shaped intraday volume factor
    t_intraday = np.linspace(-1.0, 1.0, bars_per_session)
    u_curve = 1.0 + 1.2 * (t_intraday ** 2)  # higher at boundaries
    volume_profile = np.tile(u_curve, num_sessions)

    # Annualization factor: 252 days * 78 bars = 19,656 bars/yr
    dt = 1.0 / (252.0 * bars_per_session)
    base_sigma_bar = base_annual_volatility * np.sqrt(dt)

    dfs: List[pd.DataFrame] = []

    # Latent market factor
    market_innovations = rng.standard_normal(total_bars)

    for a_idx, asset_id in enumerate(asset_ids):
        # Asset-specific beta and volatility scaling
        beta = 0.7 + 0.6 * (a_idx / max(1, num_assets - 1))
        asset_vol_scale = 0.8 + 0.4 * rng.uniform()

        # Regime-switching Markov volatility: Regime 0 (Low Vol, 0.7x), Regime 1 (High Vol, 1.8x)
        regime = 0
        regimes = np.zeros(total_bars)
        for t in range(total_bars):
            if rng.uniform() < regime_switch_prob:
                regime = 1 - regime
            regimes[t] = regime

        vol_multipliers = np.where(regimes == 1, 1.8, 0.8)
        bar_sigma = base_sigma_bar * asset_vol_scale * vol_multipliers

        # Asset price generation with AR(1) and latent momentum signal
        returns = np.zeros(total_bars)
        shocks = rng.standard_normal(total_bars)
        prev_ret = 0.0

        for t in range(total_bars):
            # AR(1) + Market Factor + Asset Shock + Controlled Lead-Lag Signal
            market_component = beta * market_innovations[t] * base_sigma_bar
            idio_shock = shocks[t] * bar_sigma[t]

            # Predictable component: weak mean-reversion on 1-bar return plus cross-sectional drift
            predictable = -autocorrelation_coeff * prev_ret + (signal_strength * (a_idx - num_assets / 2.0) / num_assets) * bar_sigma[t]
            ret_t = market_component + idio_shock + predictable
            returns[t] = ret_t
            prev_ret = ret_t

        # Generate price paths from initial price ~ 100
        initial_price = 100.0 + 10.0 * a_idx
        log_prices = np.log(initial_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)

        # Open prices: close of previous bar + small micro-noise
        open_prices = np.zeros(total_bars)
        open_prices[0] = initial_price
        open_prices[1:] = close_prices[:-1]

        # Generate realistic High and Low consistent with bar volatility
        high_spread = np.abs(rng.standard_normal(total_bars)) * bar_sigma * close_prices * 0.8
        low_spread = np.abs(rng.standard_normal(total_bars)) * bar_sigma * close_prices * 0.8

        high_prices = np.maximum(open_prices, close_prices) + high_spread
        low_prices = np.minimum(open_prices, close_prices) - low_spread

        # Enforce positive lower bound
        low_prices = np.maximum(low_prices, 0.01)

        # Traded volume: log-normal around base volume modulated by U-shaped curve and return size
        base_volume = 40000.0 * (1.0 + 0.5 * (a_idx % 3))
        volume = base_volume * volume_profile * np.exp(0.3 * np.abs(returns) / bar_sigma + 0.2 * rng.standard_normal(total_bars))
        volume = np.maximum(volume, 100.0)

        asset_df = pd.DataFrame({
            "timestamp": timestamps,
            "asset_id": asset_id,
            "open": np.round(open_prices, 4),
            "high": np.round(high_prices, 4),
            "low": np.round(low_prices, 4),
            "close": np.round(close_prices, 4),
            "volume": np.round(volume, 0),
        })
        dfs.append(asset_df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)
    return full_df
