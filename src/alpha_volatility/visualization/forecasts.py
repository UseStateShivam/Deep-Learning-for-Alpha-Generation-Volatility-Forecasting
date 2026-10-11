"""
Forecast Visualization
Plots predicted vs realized returns, residual distributions, and volatility forecasts.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predicted_vs_realized_returns(
    df: pd.DataFrame,
    pred_col: str = "pred_return_mean",
    target_col: str = "target_return",
    output_path: Optional[str | Path] = None,
    title: str = "Predicted vs Realized Forward Returns",
) -> plt.Figure:
    """Scatter plot and regression line for return forecasts."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    valid = df.dropna(subset=[pred_col, target_col])

    x = valid[pred_col].values
    y = valid[target_col].values

    ax.scatter(x, y, alpha=0.3, edgecolors="none", s=20, color="#1f77b4", label="Observations")

    # Trend line
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, m * x_line + b, color="#d62728", lw=2, label=f"Fit (Slope={m:.2f})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Expected Return", fontsize=10)
    ax.set_ylabel("Realized Forward Return", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_volatility_calibration(
    calibration_data: dict,
    output_path: Optional[str | Path] = None,
    title: str = "Volatility Forecast Calibration",
) -> plt.Figure:
    """Plots predicted volatility bucket vs realized volatility."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    x = calibration_data["mean_predicted_vol"]
    y = calibration_data["mean_realized_vol"]

    ax.plot(x, y, marker="o", color="#2ca02c", lw=2, label="Model Calibration")
    # 45-degree reference line
    min_v = min(np.nanmin(x), np.nanmin(y))
    max_v = max(np.nanmax(x), np.nanmax(y))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray", label="Perfect Calibration")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Realized Volatility", fontsize=10)
    ax.set_ylabel("Realized Volatility", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig
