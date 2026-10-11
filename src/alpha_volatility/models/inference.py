"""
Inference Module
Unified predictor executing batched inference across neural and tree models.
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch

from ..reproducibility import get_device


class UnifiedPredictor:
    """Provides inference dispatch for TCN, Transformer, and LightGBM models."""

    def __init__(
        self,
        model: Optional[Any] = None,
        model_type: str = "tcn",
        feature_names: List[str] = None,
        sequence_length: int = 24,
        device: str = "auto",
    ):
        if feature_names is None:
            feature_names = []
        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        self.device = get_device(device) if self.model_type in ["tcn", "transformer"] else "cpu"

        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes point-in-time predictions and appends prediction columns.

        Returns DataFrame aligned with input rows (observations with insufficient
        history < sequence_length are filled with neutral defaults).
        """
        out = df.copy()

        if self.model_type == "lightgbm":
            preds = self.model.predict(out)
            for k, v in preds.items():
                out[k] = v
            return out

        # For Neural Sequence Models (TCN, Transformer)
        # Construct sequences per asset
        pred_mu = np.full(len(out), 0.0)
        pred_scale = np.full(len(out), 0.01)
        pred_df = np.full(len(out), 30.0)
        pred_vol = np.full(len(out), 0.01)

        # Map row index to asset series
        for _asset, asset_df in out.groupby("asset_id"):
            asset_indices = asset_df.index.values
            feat_vals = asset_df[self.feature_names].values
            n_rows = len(feat_vals)

            if n_rows < self.sequence_length:
                continue

            # Construct rolling windows of shape [N - L + 1, L, F]
            num_windows = n_rows - self.sequence_length + 1
            windows = np.zeros((num_windows, self.sequence_length, len(self.feature_names)), dtype=np.float32)
            for w in range(num_windows):
                windows[w] = feat_vals[w : w + self.sequence_length]

            # Batched inference
            batch_size = 128
            mus, sigmas, nus, vols = [], [], [], []

            with torch.no_grad():
                for b_start in range(0, num_windows, batch_size):
                    b_end = min(b_start + batch_size, num_windows)
                    x_t = torch.tensor(windows[b_start:b_end], dtype=torch.float32, device=self.device)
                    out_dict = self.model(x_t)

                    mus.append(out_dict["pred_return_mean"].cpu().numpy())
                    sigmas.append(out_dict["pred_return_scale"].cpu().numpy())
                    nus.append(out_dict["pred_return_df"].cpu().numpy())
                    vols.append(out_dict["pred_volatility"].cpu().numpy())

            valid_row_indices = asset_indices[self.sequence_length - 1 :]
            pred_mu[valid_row_indices] = np.concatenate(mus)
            pred_scale[valid_row_indices] = np.concatenate(sigmas)
            pred_df[valid_row_indices] = np.concatenate(nus)
            pred_vol[valid_row_indices] = np.concatenate(vols)

        out["pred_return_mean"] = pred_mu
        out["pred_return_scale"] = pred_scale
        out["pred_return_df"] = pred_df
        out["pred_volatility"] = pred_vol
        return out
