"""
Walk-Forward Splitting and Purged Cross-Validation
Implements strict point-in-time walk-forward evaluation, purged splits, and holdout isolation.
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .embargo import apply_embargo
from .purging import purge_overlapping_samples


@dataclass
class SplitFold:
    """Detailed metadata and index arrays for a cross-validation fold."""
    fold_id: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    initial_train_count: int
    purged_count: int
    embargoed_count: int
    final_train_count: int
    val_count: int


class PurgedWalkForwardCV:
    """
    Constructs expanding-window chronological walk-forward splits with purging and embargo.
    Strictly reserves a final untouched holdout test period.
    """

    def __init__(
        self,
        n_splits: int = 4,
        holdout_ratio: float = 0.20,
        embargo_bars: int = 12,
        min_train_ratio: float = 0.40,
        window_type: str = "expanding",
    ):
        self.n_splits = n_splits
        self.holdout_ratio = holdout_ratio
        self.embargo_bars = embargo_bars
        self.min_train_ratio = min_train_ratio
        self.window_type = window_type

    def split_holdout(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into research development data and strictly untouched final test data.
        """
        sorted_df = df.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)
        unique_times = sorted_df["timestamp"].drop_duplicates().sort_values().tolist()
        n_times = len(unique_times)

        split_idx = int(n_times * (1.0 - self.holdout_ratio))
        cutoff_time = unique_times[split_idx]

        dev_df = sorted_df[sorted_df["timestamp"] < cutoff_time].copy()
        test_df = sorted_df[sorted_df["timestamp"] >= cutoff_time].copy()

        # Purge dev_df of any observations whose forward label crosses into test_df
        if "label_info_end" in dev_df.columns:
            dev_df, _ = purge_overlapping_samples(dev_df, test_df)

        return dev_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def generate_folds(self, dev_df: pd.DataFrame) -> List[SplitFold]:
        """
        Generates purged, embargoed walk-forward evaluation folds from the development data.
        """
        unique_times = dev_df["timestamp"].drop_duplicates().sort_values().tolist()
        n_times = len(unique_times)

        min_train_idx = int(n_times * self.min_train_ratio)
        eval_times = unique_times[min_train_idx:]
        val_step = len(eval_times) // self.n_splits

        folds: List[SplitFold] = []

        for i in range(self.n_splits):
            fold_id = f"fold_{i+1:02d}"
            val_start_idx = min_train_idx + i * val_step
            val_end_idx = min_train_idx + (i + 1) * val_step if i < self.n_splits - 1 else len(unique_times)

            val_start_time = unique_times[val_start_idx]
            val_end_time = unique_times[val_end_idx - 1]

            val_df = dev_df[
                (dev_df["timestamp"] >= val_start_time) & (dev_df["timestamp"] <= val_end_time)
            ].copy()

            if self.window_type == "rolling" and i > 0:
                train_start_time = unique_times[i * val_step]
            else:
                train_start_time = unique_times[0]

            train_df_raw = dev_df[
                (dev_df["timestamp"] >= train_start_time) & (dev_df["timestamp"] < val_start_time)
            ].copy()

            initial_train_count = len(train_df_raw)

            # Purge overlapping samples
            purged_train_df, purged_count = purge_overlapping_samples(train_df_raw, val_df)

            # Apply embargo
            embargoed_train_df, embargoed_count = apply_embargo(
                purged_train_df, val_df, embargo_bars=self.embargo_bars
            )

            train_start = embargoed_train_df["timestamp"].min() if len(embargoed_train_df) > 0 else pd.NaT
            train_end = embargoed_train_df["timestamp"].max() if len(embargoed_train_df) > 0 else pd.NaT
            val_start = val_df["timestamp"].min() if len(val_df) > 0 else pd.NaT
            val_end = val_df["timestamp"].max() if len(val_df) > 0 else pd.NaT

            split_fold = SplitFold(
                fold_id=fold_id,
                train_df=embargoed_train_df.reset_index(drop=True),
                val_df=val_df.reset_index(drop=True),
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                initial_train_count=initial_train_count,
                purged_count=purged_count,
                embargoed_count=embargoed_count,
                final_train_count=len(embargoed_train_df),
                val_count=len(val_df),
            )
            folds.append(split_fold)

        return folds
