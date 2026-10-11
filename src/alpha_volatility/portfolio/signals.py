"""
Signal Processing Module
Extracts, winsorizes, and cross-sectionally standardizes alpha signals for portfolio construction.
"""

import numpy as np
import pandas as pd


def process_alpha_signal(
    df: pd.DataFrame,
    signal_col: str = "pred_return_mean",
    standardize_cross_sectional: bool = False,
    clip_quantile: float = 0.01,
) -> pd.DataFrame:
    """
    Cleans and standardizes raw predicted return scores point-in-time.

    Args:
        df: DataFrame containing 'timestamp', 'asset_id', and signal_col.
        signal_col: Column name of raw predictions.
        standardize_cross_sectional: Whether to z-score signals cross-sectionally per timestamp.
        clip_quantile: Winsorization threshold for extreme signal outliers.

    Returns:
        DataFrame with processed 'alpha_signal' column.
    """
    out = df.copy()
    if signal_col not in out.columns:
        raise KeyError(f"Signal column '{signal_col}' not found in predictions DataFrame.")

    # Winsorize extremes
    low_bound = out[signal_col].quantile(clip_quantile)
    high_bound = out[signal_col].quantile(1.0 - clip_quantile)
    out["alpha_signal"] = np.clip(out[signal_col], low_bound, high_bound)

    if standardize_cross_sectional:
        def zscore_group(g: pd.Series) -> pd.Series:
            std = g.std()
            if std < 1e-8 or len(g) <= 1:
                return pd.Series(0.0, index=g.index)
            return (g - g.mean()) / std

        out["alpha_signal"] = out.groupby("timestamp")["alpha_signal"].transform(zscore_group)

    return out
