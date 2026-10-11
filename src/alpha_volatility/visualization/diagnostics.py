"""
Diagnostic Visualization
Plots loss curves, feature correlation heatmaps, and IC distributions.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_curves(
    history: List[dict],
    output_path: Optional[str | Path] = None,
    title: str = "Training and Validation Loss Trajectories",
) -> plt.Figure:
    """Plots training and validation loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", np.nan) for h in history]
    val_loss = [h.get("val_loss", np.nan) for h in history]

    ax.plot(epochs, train_loss, marker="o", label="Training Loss", color="#1f77b4")
    if any(~np.isnan(val_loss)):
        ax.plot(epochs, val_loss, marker="s", label="Validation Loss", color="#ff7f0e")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Multi-Task Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_ic_distribution_and_series(
    ic_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Information Coefficient Diagnostics",
) -> plt.Figure:
    """Two-panel plot: IC time-series with rolling mean and IC histogram."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

    # 1. Time Series
    ax1.plot(ic_df["timestamp"], ic_df["rank_ic"], color="#1f77b4", alpha=0.4, label="Timestamp Rank IC")
    roll_mean = ic_df["rank_ic"].rolling(30, min_periods=5).mean()
    ax1.plot(ic_df["timestamp"], roll_mean, color="#d62728", lw=2, label="30-Period Rolling Mean")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax1.set_title(f"{title} - Time Series", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rank IC", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Histogram
    ax2.hist(ic_df["rank_ic"].dropna(), bins=30, color="#2ca02c", edgecolor="white", alpha=0.8)
    mean_val = float(ic_df["rank_ic"].mean())
    ax2.axvline(mean_val, color="#d62728", linestyle="--", lw=2, label=f"Mean = {mean_val:.4f}")
    ax2.axvline(0, color="black", linestyle=":", alpha=0.8)
    ax2.set_title("Rank IC Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Rank IC", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Generates correlation heatmap across key feature subsets."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sub_cols = feature_cols[:15]  # Display prominent subset for legibility
    corr = df[sub_cols].corr()

    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)

    ax.set_xticks(range(len(sub_cols)))
    ax.set_yticks(range(len(sub_cols)))
    ax.set_xticklabels(sub_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(sub_cols, fontsize=8)
    ax.set_title("Point-in-Time Feature Correlation Matrix", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig
