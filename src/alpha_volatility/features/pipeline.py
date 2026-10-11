"""
Feature Pipeline Module
Orchestrates technical, temporal, and cross-sectional feature generation,
and provides fold-specific normalization without information leakage.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .cross_sectional import compute_cross_sectional_features
from .technical import compute_technical_features
from .temporal import compute_temporal_features

logger = logging.getLogger("alpha_volatility.features.pipeline")

# Canonical feature columns in exact fixed order
CORE_FEATURE_COLUMNS: List[str] = [
    # Price / Return
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "log_return_12",
    "high_low_range",
    "open_close_return",
    "price_dev_12",
    # Volatility
    "realized_vol_6",
    "realized_vol_12",
    "realized_vol_36",
    "parkinson_vol_12",
    "vol_ratio_6_36",
    # Volume
    "log_volume",
    "volume_zscore_12",
    "relative_volume_12",
    "pv_interaction",
    # Momentum
    "momentum_6",
    "momentum_24",
    "trend_consistency_12",
    # Temporal
    "sin_time_of_day",
    "cos_time_of_day",
    "session_progress",
    "day_of_week_sin",
    "day_of_week_cos",
    # Cross-Sectional
    "cs_rank_log_return_1",
    "cs_rank_momentum_6",
    "cs_rank_realized_vol_12",
]


class FeaturePipeline:
    """Manages end-to-end feature calculation and fold-isolated normalization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_columns = CORE_FEATURE_COLUMNS.copy()

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all features point-in-time.
        Guarantees that observations at time t use only data <= t.
        """
        logger.info("Computing technical features...")
        feat_df = compute_technical_features(df)

        logger.info("Computing temporal features...")
        feat_df = compute_temporal_features(feat_df)

        logger.info("Computing cross-sectional rank features...")
        feat_df = compute_cross_sectional_features(
            feat_df, features_to_rank=["log_return_1", "momentum_6", "realized_vol_12"]
        )

        # Ensure all core features exist and fill any initial warmup NaNs with 0.0
        for col in self.feature_columns:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
            else:
                feat_df[col] = feat_df[col].fillna(0.0)

        # Sort strictly by timestamp and asset_id
        feat_df = feat_df.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)
        return feat_df

    @staticmethod
    def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str], method: str = "standard") -> Any:
        """
        Fits a scaler strictly on training fold data.
        NEVER fits across validation or test periods!
        """
        X_train = train_df[feature_cols].values
        if method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(X_train)
        return scaler

    @staticmethod
    def transform_features(
        df: pd.DataFrame,
        scaler: Any,
        feature_cols: List[str],
        clip_sigma: float = 5.0,
    ) -> pd.DataFrame:
        """
        Transforms feature columns using a fitted scaler and clips extreme outliers.
        """
        out = df.copy()
        X = out[feature_cols].values
        X_scaled = scaler.transform(X)
        if clip_sigma is not None and clip_sigma > 0:
            X_scaled = np.clip(X_scaled, -clip_sigma, clip_sigma)
        out[feature_cols] = X_scaled
        return out
