"""
Labels module for future returns, realized volatility, and information interval alignment.
"""

from .alignment import attach_label_information_intervals, generate_all_labels
from .returns import compute_future_returns
from .volatility import compute_future_realized_volatility

__all__ = [
    "compute_future_returns",
    "compute_future_realized_volatility",
    "attach_label_information_intervals",
    "generate_all_labels",
]
