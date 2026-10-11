"""
Benchmark Baselines Module
Implements non-parametric quantitative baselines: Zero Return and Rolling Volatility Persistence.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster


class ZeroReturnBaseline(BaseForecaster):
    """Predicts a constant zero expected return and unconditional historical sample volatility."""

    def __init__(self, default_vol: float = 0.01):
        self.default_vol = default_vol
        self.fitted_vol = default_vol

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "ZeroReturnBaseline":
        if "target_realized_vol" in train_df.columns:
            self.fitted_vol = float(train_df["target_realized_vol"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        n = len(df)
        return {
            "pred_return_mean": np.zeros(n),
            "pred_return_scale": np.full(n, self.fitted_vol),
            "pred_return_df": np.full(n, 30.0),
            "pred_volatility": np.full(n, self.fitted_vol),
        }

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> "ZeroReturnBaseline":
        return self


class HistoricalVolatilityPersistenceBaseline(BaseForecaster):
    """
    Volatility persistence benchmark: forecasts forward realized volatility
    using the most recent backward realized volatility observation (e.g. realized_vol_12).
    """

    def __init__(self, feature_col: str = "realized_vol_12"):
        self.feature_col = feature_col

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "HistoricalVolatilityPersistenceBaseline":
        return self

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        n = len(df)
        if self.feature_col in df.columns:
            vol = np.maximum(df[self.feature_col].values, 1e-6)
        else:
            vol = np.full(n, 0.01)

        return {
            "pred_return_mean": np.zeros(n),
            "pred_return_scale": vol,
            "pred_return_df": np.full(n, 30.0),
            "pred_volatility": vol,
        }

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> "HistoricalVolatilityPersistenceBaseline":
        return self
