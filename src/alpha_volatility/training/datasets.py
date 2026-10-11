"""
Time-Series Sequence Datasets
Efficient PyTorch Dataset constructing sliding temporal windows (L, F) without full memory replication.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FinancialSequenceDataset(Dataset):
    """
    Constructs sequence windows [t - seq_len + 1, t] and aligns forward targets [y_ret, y_vol] at time t.
    Window construction is isolated per asset to prevent sequence history crossing asset boundaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int = 24,
        return_col: str = "target_return",
        vol_col: str = "target_realized_vol",
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.return_col = return_col
        self.vol_col = vol_col

        self.samples: List[Tuple[np.ndarray, float, float]] = []
        self._build_index(df)

    def _build_index(self, df: pd.DataFrame) -> None:
        """Slices sliding windows per asset."""
        for _, asset_df in df.groupby("asset_id"):
            sorted_adf = asset_df.sort_values("timestamp")
            feats = sorted_adf[self.feature_cols].values.astype(np.float32)
            y_ret = sorted_adf[self.return_col].values.astype(np.float32)
            y_vol = sorted_adf[self.vol_col].values.astype(np.float32)

            n_rows = len(feats)
            if n_rows < self.seq_len:
                continue

            for t in range(self.seq_len - 1, n_rows):
                # Window ends at index t, using past seq_len observations
                window = feats[t - self.seq_len + 1 : t + 1]
                target_r = y_ret[t]
                target_v = y_vol[t]

                # Only include if targets are non-null
                if not (np.isnan(target_r) or np.isnan(target_v)):
                    self.samples.append((window, float(target_r), float(target_v)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, ret_target, vol_target = self.samples[idx]
        return (
            torch.from_numpy(window),
            torch.tensor(ret_target, dtype=torch.float32),
            torch.tensor(vol_target, dtype=torch.float32),
        )
