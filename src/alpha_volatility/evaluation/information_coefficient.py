"""
Information Coefficient (IC) Module
Computes point-in-time cross-sectional Pearson IC and Spearman Rank IC across the asset universe.
"""

from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_cross_sectional_ic_series(
    df: pd.DataFrame,
    pred_col: str = "pred_return_mean",
    target_col: str = "target_return",
    min_assets: int = 3,
) -> pd.DataFrame:
    """
    Computes timestamp-level cross-sectional Pearson IC and Spearman Rank IC.

    Args:
        df: DataFrame containing 'timestamp', 'asset_id', pred_col, target_col.
        pred_col: Name of predicted alpha signal column.
        target_col: Name of realized forward return column.
        min_assets: Minimum number of eligible assets required at timestamp t.

    Returns:
        DataFrame indexed by timestamp with columns ['ic', 'rank_ic', 'num_assets'].
    """
    records = []

    for ts, group in df.groupby("timestamp"):
        valid = group.dropna(subset=[pred_col, target_col])
        n = len(valid)
        if n < min_assets:
            continue

        p_vals = valid[pred_col].values
        t_vals = valid[target_col].values

        # Check for non-constant arrays
        if np.std(p_vals) < 1e-8 or np.std(t_vals) < 1e-8:
            records.append({"timestamp": ts, "ic": 0.0, "rank_ic": 0.0, "num_assets": n})
            continue

        # Pearson IC
        ic = float(np.corrcoef(p_vals, t_vals)[0, 1])

        # Spearman Rank IC
        rank_ic, _ = stats.spearmanr(p_vals, t_vals)
        if np.isnan(rank_ic):
            rank_ic = 0.0

        records.append({
            "timestamp": ts,
            "ic": ic,
            "rank_ic": float(rank_ic),
            "num_assets": n,
        })

    if not records:
        return pd.DataFrame(columns=["timestamp", "ic", "rank_ic", "num_assets"])

    res_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return res_df


def summarize_ic_metrics(ic_df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarizes distribution of timestamp-level IC and Rank IC observations.
    """
    if ic_df.empty or "ic" not in ic_df.columns:
        return {
            "mean_ic": 0.0,
            "median_ic": 0.0,
            "std_ic": 0.0,
            "ic_ir": 0.0,
            "pct_positive_ic": 0.0,
            "mean_rank_ic": 0.0,
            "median_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "rank_ic_ir": 0.0,
            "pct_positive_rank_ic": 0.0,
            "valid_timestamps": 0,
            "avg_assets_per_timestamp": 0.0,
        }

    ic = ic_df["ic"].dropna().values
    rank_ic = ic_df["rank_ic"].dropna().values

    mean_ic = float(np.mean(ic))
    std_ic = float(np.std(ic, ddof=1)) if len(ic) > 1 else 1e-6
    ic_ir = mean_ic / max(std_ic, 1e-6)

    mean_rank_ic = float(np.mean(rank_ic))
    std_rank_ic = float(np.std(rank_ic, ddof=1)) if len(rank_ic) > 1 else 1e-6
    rank_ic_ir = mean_rank_ic / max(std_rank_ic, 1e-6)

    return {
        "mean_ic": round(mean_ic, 4),
        "median_ic": round(float(np.median(ic)), 4),
        "std_ic": round(std_ic, 4),
        "ic_ir": round(ic_ir, 4),
        "pct_positive_ic": round(float(np.mean(ic > 0)), 4),
        "mean_rank_ic": round(mean_rank_ic, 4),
        "median_rank_ic": round(float(np.median(rank_ic)), 4),
        "std_rank_ic": round(std_rank_ic, 4),
        "rank_ic_ir": round(rank_ic_ir, 4),
        "pct_positive_rank_ic": round(float(np.mean(rank_ic > 0)), 4),
        "valid_timestamps": int(len(ic_df)),
        "avg_assets_per_timestamp": round(float(ic_df["num_assets"].mean()), 1),
    }
