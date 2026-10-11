"""
Evaluation module for forecast metrics, cross-sectional IC, portfolio statistics, and reporting.
"""

from .forecasting import compute_forecasting_metrics
from .information_coefficient import compute_cross_sectional_ic_series, summarize_ic_metrics
from .portfolio_metrics import compute_drawdown_series, compute_portfolio_performance_metrics
from .regime_analysis import evaluate_by_volatility_regimes
from .reports import generate_model_comparison_table, save_metrics_report
from .statistical_tests import diebold_mariano_test, stationary_block_bootstrap

__all__ = [
    "compute_forecasting_metrics",
    "compute_cross_sectional_ic_series",
    "summarize_ic_metrics",
    "compute_portfolio_performance_metrics",
    "compute_drawdown_series",
    "stationary_block_bootstrap",
    "diebold_mariano_test",
    "evaluate_by_volatility_regimes",
    "generate_model_comparison_table",
    "save_metrics_report",
]
