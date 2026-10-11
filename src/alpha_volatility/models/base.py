"""
Base Model Interface
Defines the standard contract for alpha forecasting and volatility models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class BaseForecaster(ABC):
    """Abstract base class for all forecasting estimators."""

    @abstractmethod
    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> "BaseForecaster":
        """Trains the forecaster on provided training data with optional validation set."""
        pass

    @abstractmethod
    def predict(self, data: Any) -> Dict[str, np.ndarray]:
        """
        Generates predictions for return distribution parameters and volatility.

        Returns a dictionary with keys:
          - 'pred_return_mean': expected return E[r]
          - 'pred_return_scale': dispersion / scale parameter
          - 'pred_return_df': degrees of freedom (if Student-t)
          - 'pred_volatility': point estimate of forward realized volatility
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serializes the fitted model state and metadata."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> "BaseForecaster":
        """Deserializes and reconstructs the model."""
        pass
