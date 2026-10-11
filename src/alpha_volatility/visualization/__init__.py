"""
Visualization module for forecasting diagnostics, equity curves, decile charts, and research plots.
"""

from .diagnostics import (
    plot_feature_correlation_heatmap,
    plot_ic_distribution_and_series,
    plot_loss_curves,
)
from .forecasts import plot_predicted_vs_realized_returns, plot_volatility_calibration
from .portfolios import plot_decile_returns_bar_chart, plot_equity_and_drawdown
from .report_plots import generate_all_report_figures

__all__ = [
    "plot_predicted_vs_realized_returns",
    "plot_volatility_calibration",
    "plot_loss_curves",
    "plot_ic_distribution_and_series",
    "plot_feature_correlation_heatmap",
    "plot_decile_returns_bar_chart",
    "plot_equity_and_drawdown",
    "generate_all_report_figures",
]
