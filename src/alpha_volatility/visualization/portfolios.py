"""
Portfolio Performance Visualization
Plots decile spread bar charts, cumulative gross vs net equity curves, and underwater drawdowns.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_decile_returns_bar_chart(
    decile_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Mean Realized Return by Signal Decile",
) -> plt.Figure:
    """Bar chart illustrating forward returns across deciles 1 through 10."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    deciles = decile_df["decile"].values
    returns = decile_df["mean_return"].values * 10000.0  # In basis points

    colors = ["#d62728" if d <= 3 else "#2ca02c" if d >= 8 else "#1f77b4" for d in deciles]
    bars = ax.bar(deciles, returns, color=colors, edgecolor="black", alpha=0.85)

    ax.axhline(0, color="black", linestyle="-", lw=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Quantile Decile (1 = Lowest Predicted, 10 = Highest Predicted)", fontsize=10)
    ax.set_ylabel("Mean Forward Return (Basis Points)", fontsize=10)
    ax.set_xticks(deciles)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add value labels
    for b in bars:
        height = b.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_equity_and_drawdown(
    equity_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Dollar-Neutral Decile Strategy Cumulative Performance",
) -> plt.Figure:
    """Two-panel plot: Gross vs Net Equity Curves, and Underwater Drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    # 1. Equity Curves
    cum_gross = (equity_df["cum_gross_wealth"] - 1.0) * 100.0
    cum_net = (equity_df["cum_net_wealth"] - 1.0) * 100.0

    ax1.plot(equity_df["timestamp"], cum_gross, color="#1f77b4", lw=2, label="Gross Cumulative Return (%)")
    ax1.plot(equity_df["timestamp"], cum_net, color="#2ca02c", lw=2, label="Net Cumulative Return (After Costs, %)")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Drawdowns
    wealth = equity_df["cum_net_wealth"].values
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / np.maximum(peak, 1e-8) * 100.0

    ax2.fill_between(equity_df["timestamp"], dd, 0, color="#d62728", alpha=0.4, label="Net Drawdown (%)")
    ax2.plot(equity_df["timestamp"], dd, color="#d62728", lw=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Timestamp", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig
