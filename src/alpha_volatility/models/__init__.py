"""
Models module containing deep learning (TCN, Transformer) and tabular (LightGBM) architectures.
"""

from .base import BaseForecaster
from .baselines import HistoricalVolatilityPersistenceBaseline, ZeroReturnBaseline
from .calibration import (
    compute_predictive_quantiles,
    compute_prob_positive_return,
    compute_volatility_calibration_buckets,
    evaluate_interval_coverage,
)
from .inference import UnifiedPredictor
from .lightgbm_model import LightGBMForecaster
from .losses import MultiTaskLoss, gaussian_nll, qlike_loss, student_t_nll
from .tcn import TCNForecaster, TemporalBlock, TemporalConvNet
from .transformer import TransformerForecaster

__all__ = [
    "BaseForecaster",
    "student_t_nll",
    "gaussian_nll",
    "qlike_loss",
    "MultiTaskLoss",
    "TCNForecaster",
    "TemporalConvNet",
    "TemporalBlock",
    "TransformerForecaster",
    "LightGBMForecaster",
    "ZeroReturnBaseline",
    "HistoricalVolatilityPersistenceBaseline",
    "compute_prob_positive_return",
    "compute_predictive_quantiles",
    "evaluate_interval_coverage",
    "compute_volatility_calibration_buckets",
    "UnifiedPredictor",
]
