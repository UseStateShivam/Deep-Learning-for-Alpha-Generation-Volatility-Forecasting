"""
Portfolio module for signal processing, decile allocation, transaction costs, and backtesting.
"""

from .backtest import BacktestResult, DecilePortfolioBacktester
from .constraints import apply_position_limits
from .construction import assign_decile_quantiles, construct_dollar_neutral_weights
from .costs import TransactionCostModel
from .execution import apply_execution_lag
from .signals import process_alpha_signal

__all__ = [
    "process_alpha_signal",
    "assign_decile_quantiles",
    "construct_dollar_neutral_weights",
    "apply_position_limits",
    "TransactionCostModel",
    "apply_execution_lag",
    "DecilePortfolioBacktester",
    "BacktestResult",
]
