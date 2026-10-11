"""
Portfolio Metrics Module
Calculates risk-adjusted returns, Sharpe, Sortino, drawdowns, and transaction cost drag.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def compute_drawdown_series(wealth_index: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes percentage underwater drawdown series and maximum drawdown.

    Args:
        wealth_index: Array of cumulative portfolio equity (starting at 1.0).

    Returns:
        (drawdown_series, max_drawdown)
    """
    peak = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - peak) / np.maximum(peak, 1e-8)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    return drawdowns, abs(max_dd)


def compute_portfolio_performance_metrics(
    returns: np.ndarray,
    bars_per_year: int = 19656,  # 252 days * 78 bars/day
    turnover_series: Optional[np.ndarray] = None,
    cost_series: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculates institutional-grade backtesting summary statistics.

    Args:
        returns: Array of period portfolio returns.
        bars_per_year: Annualization factor (e.g. 19656 for 5m intraday bars).
        turnover_series: Optional array of per-period portfolio turnover.
        cost_series: Optional array of per-period transaction costs.

    Returns:
        Dictionary of performance metrics.
    """
    if len(returns) == 0:
        return {}

    # Wealth index
    wealth = np.cumprod(1.0 + returns)
    total_cum_ret = float(wealth[-1] - 1.0)

    n_bars = len(returns)
    float(bars_per_year) / float(n_bars)
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if n_bars > 1 else 1e-6

    # Annualized return and volatility
    ann_return = mean_ret * bars_per_year
    ann_vol = std_ret * np.sqrt(bars_per_year)

    # Sharpe Ratio
    sharpe = ann_return / max(ann_vol, 1e-6)

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0.0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(bars_per_year)
        sortino = ann_return / max(downside_std, 1e-6)
    else:
        sortino = sharpe

    # Drawdowns
    _, max_drawdown = compute_drawdown_series(wealth)
    calmar = ann_return / max(max_drawdown, 1e-6)

    # Win rate and payoff ratio
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    win_rate = float(len(wins) / n_bars)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 1e-6
    win_loss_ratio = avg_win / avg_loss

    metrics = {
        "cumulative_return": round(total_cum_ret, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "win_loss_ratio": round(win_loss_ratio, 4),
        "total_bars": n_bars,
    }

    if turnover_series is not None and len(turnover_series) > 0:
        metrics["mean_turnover"] = round(float(np.mean(turnover_series)), 4)
        metrics["total_turnover"] = round(float(np.sum(turnover_series)), 4)

    if cost_series is not None and len(cost_series) > 0:
        metrics["total_cost_drag"] = round(float(np.sum(cost_series)), 6)

    return metrics
