"""
Master Report Figure Generator
Coordinates generation and serialization of all required research figures.
"""

from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..models.calibration import compute_volatility_calibration_buckets
from .diagnostics import plot_feature_correlation_heatmap, plot_ic_distribution_and_series
from .forecasts import plot_predicted_vs_realized_returns, plot_volatility_calibration
from .portfolios import plot_decile_returns_bar_chart, plot_equity_and_drawdown


def generate_all_report_figures(
    predictions_df: pd.DataFrame,
    backtest_result: Optional[Any] = None,
    ic_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    output_dir: str | Path = "artifacts/figures",
) -> List[Path]:
    """
    Generates and saves all key research figures required by Section 23.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # 1. Predicted vs Realized Returns
    p1 = out_dir / "predicted_vs_realized_returns.png"
    plot_predicted_vs_realized_returns(predictions_df, output_path=p1)
    saved_paths.append(p1)
    plt.close("all")

    # 2. Volatility Calibration
    if "target_realized_vol" in predictions_df.columns and "pred_volatility" in predictions_df.columns:
        valid_v = predictions_df.dropna(subset=["target_realized_vol", "pred_volatility"])
        calib_data = compute_volatility_calibration_buckets(
            realized_vol=valid_v["target_realized_vol"].values,
            predicted_vol=valid_v["pred_volatility"].values,
            num_bins=5,
        )
        p2 = out_dir / "volatility_calibration.png"
        plot_volatility_calibration(calib_data, output_path=p2)
        saved_paths.append(p2)
        plt.close("all")

    # 3. IC Series & Distribution
    if ic_df is not None and not ic_df.empty:
        p3 = out_dir / "ic_diagnostics.png"
        plot_ic_distribution_and_series(ic_df, output_path=p3)
        saved_paths.append(p3)
        plt.close("all")

    # 4. Decile Bar Chart & Equity Curves
    if backtest_result is not None:
        p4 = out_dir / "decile_returns.png"
        plot_decile_returns_bar_chart(backtest_result.decile_returns, output_path=p4)
        saved_paths.append(p4)
        plt.close("all")

        p5 = out_dir / "equity_and_drawdown.png"
        plot_equity_and_drawdown(backtest_result.equity_curve, output_path=p5)
        saved_paths.append(p5)
        plt.close("all")

    # 5. Feature Correlation Heatmap
    if feature_cols:
        p6 = out_dir / "feature_correlations.png"
        plot_feature_correlation_heatmap(predictions_df, feature_cols, output_path=p6)
        saved_paths.append(p6)
        plt.close("all")

    return saved_paths
