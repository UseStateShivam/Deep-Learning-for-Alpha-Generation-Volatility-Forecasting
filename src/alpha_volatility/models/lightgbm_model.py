"""
LightGBM Baseline Forecaster
Implements gradient-boosted tree baseline models for return and realized volatility forecasting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM tabular baseline with dual models:
      - Return Regressor: forecasts forward expected return E[r]
      - Volatility Regressor: forecasts forward realized volatility
    """

    def __init__(
        self,
        feature_names: List[str],
        params: Optional[Dict[str, Any]] = None,
        return_col: str = "target_return",
        vol_col: str = "target_realized_vol",
    ):
        self.feature_names = feature_names
        self.return_col = return_col
        self.vol_col = vol_col
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "max_depth": 6,
            "num_leaves": 31,
            "learning_rate": 0.03,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        self.model_ret: Optional[lgb.LGBMRegressor] = None
        self.model_vol: Optional[lgb.LGBMRegressor] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "LightGBMForecaster":
        """
        Fits return and volatility models using training fold data and optional validation early stopping.
        """
        X_train = train_df[self.feature_names].values
        y_ret_train = train_df[self.return_col].values
        y_vol_train = train_df[self.vol_col].values

        fit_params = self.params.copy()
        eval_set_ret = None
        eval_set_vol = None
        callbacks = []

        if val_df is not None and not val_df.empty:
            X_val = val_df[self.feature_names].values
            y_ret_val = val_df[self.return_col].values
            y_vol_val = val_df[self.vol_col].values
            eval_set_ret = [(X_val, y_ret_val)]
            eval_set_vol = [(X_val, y_vol_val)]
            stopping_rounds = fit_params.pop("early_stopping_rounds", 15)
            callbacks = [lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
        else:
            fit_params.pop("early_stopping_rounds", None)
            fit_params.pop("early_stopping_round", None)

        # 1. Fit Return Model
        self.model_ret = lgb.LGBMRegressor(**fit_params)
        self.model_ret.fit(
            X_train,
            y_ret_train,
            eval_set=eval_set_ret,
            callbacks=callbacks,
        )

        # 2. Fit Volatility Model
        self.model_vol = lgb.LGBMRegressor(**fit_params)
        self.model_vol.fit(
            X_train,
            y_vol_train,
            eval_set=eval_set_vol,
            callbacks=callbacks,
        )

        return self

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generates predictions for return and volatility.
        """
        if self.model_ret is None or self.model_vol is None:
            raise RuntimeError("Models must be fitted before predict() is called.")

        X = df[self.feature_names].values
        pred_ret = self.model_ret.predict(X)
        pred_vol = np.maximum(self.model_vol.predict(X), 1e-6)

        # For LightGBM baseline, return scale can be proxied by predicted volatility
        return {
            "pred_return_mean": pred_ret,
            "pred_return_scale": pred_vol,
            "pred_return_df": np.full(len(pred_ret), 30.0),
            "pred_volatility": pred_vol,
        }

    def get_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """Returns gain-based feature importance dictionary for both heads."""
        if self.model_ret is None or self.model_vol is None:
            return {}

        ret_imp = self.model_ret.booster_.feature_importance(importance_type="gain")
        vol_imp = self.model_vol.booster_.feature_importance(importance_type="gain")

        return {
            "return_task": {
                name: float(imp) for name, imp in zip(self.feature_names, ret_imp, strict=False)
            },
            "volatility_task": {
                name: float(imp) for name, imp in zip(self.feature_names, vol_imp, strict=False)
            },
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_ret": self.model_ret,
                "model_vol": self.model_vol,
                "feature_names": self.feature_names,
                "params": self.params,
            },
            p,
        )

    def load(self, path: str | Path) -> "LightGBMForecaster":
        p = Path(path)
        data = joblib.load(p)
        self.model_ret = data["model_ret"]
        self.model_vol = data["model_vol"]
        self.feature_names = data["feature_names"]
        self.params = data["params"]
        return self
