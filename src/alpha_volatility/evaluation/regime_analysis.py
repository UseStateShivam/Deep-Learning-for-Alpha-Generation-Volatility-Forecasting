"""
Regime Analysis Module
Segments out-of-sample forecast accuracy and IC metrics across historical volatility regimes.
"""

from typing import Dict

import pandas as pd

from .forecasting import compute_forecasting_metrics
from .information_coefficient import compute_cross_sectional_ic_series, summarize_ic_metrics


def evaluate_by_volatility_regimes(
    predictions_df: pd.DataFrame,
    vol_feature_col: str = "realized_vol_12",
    num_regimes: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Partitions observations into volatility regimes (e.g. Low, Medium, High)
    and computes IC and forecasting metrics within each regime.

    Args:
        predictions_df: DataFrame containing predictions, targets, and volatility feature.
        vol_feature_col: Historical volatility indicator used strictly point-in-time.
        num_regimes: Number of regime buckets (3 = Tertiles: Low, Medium, High).

    Returns:
        Dictionary mapping regime names to their respective metric summaries.
    """
    if vol_feature_col not in predictions_df.columns:
        return {}

    labels = ["low_vol", "medium_vol", "high_vol"] if num_regimes == 3 else [f"regime_{i+1}" for i in range(num_regimes)]
    df = predictions_df.copy()

    # Calculate regime cuts strictly from historical vol feature
    df["regime"] = pd.qcut(df[vol_feature_col], q=num_regimes, labels=labels, duplicates="drop")

    out: Dict[str, Dict[str, float]] = {}

    for regime_name, group in df.groupby("regime", observed=True):
        ic_series = compute_cross_sectional_ic_series(group)
        ic_summary = summarize_ic_metrics(ic_series)

        fc_summary = compute_forecasting_metrics(
            y_true_ret=group["target_return"].values,
            y_pred_ret=group["pred_return_mean"].values,
            y_true_vol=group["target_realized_vol"].values,
            y_pred_vol=group["pred_volatility"].values,
        )

        out[str(regime_name)] = {
            "num_samples": int(len(group)),
            "mean_ic": ic_summary.get("mean_ic", 0.0),
            "mean_rank_ic": ic_summary.get("mean_rank_ic", 0.0),
            "return_rmse": fc_summary.get("return_rmse", 0.0),
            "volatility_qlike": fc_summary.get("volatility_qlike", 0.0),
        }

    return out
