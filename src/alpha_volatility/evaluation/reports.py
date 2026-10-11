"""
Model Comparison and Evaluation Reports
Constructs unified cross-model comparison tables and serializes structured metric summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def generate_model_comparison_table(
    model_evaluations: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Constructs a comparative summary DataFrame across all evaluated models and baselines.

    Args:
        model_evaluations: Dictionary mapping model names to metric dictionaries.

    Returns:
        Structured pandas DataFrame comparing models.
    """
    rows = []
    for model_name, metrics in model_evaluations.items():
        row = {
            "Model": model_name.upper(),
            "Return RMSE": metrics.get("return_rmse", np.nan),
            "Return MAE": metrics.get("return_mae", np.nan),
            "Directional Acc": metrics.get("return_directional_accuracy", np.nan),
            "Mean IC": metrics.get("mean_ic", np.nan),
            "Rank IC": metrics.get("mean_rank_ic", np.nan),
            "IC IR": metrics.get("ic_ir", np.nan),
            "Vol RMSE": metrics.get("volatility_rmse", np.nan),
            "Vol QLIKE": metrics.get("volatility_qlike", np.nan),
            "Sharpe (Gross)": metrics.get("sharpe_ratio_gross", np.nan),
            "Sharpe (Net)": metrics.get("sharpe_ratio_net", np.nan),
            "Max Drawdown": metrics.get("max_drawdown", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_metrics_report(
    comparison_df: pd.DataFrame,
    detailed_metrics: Dict[str, Any],
    output_dir: str | Path = "artifacts/metrics",
) -> Tuple[Path, Path]:
    """
    Saves model comparison table as CSV and detailed metrics dictionary as JSON.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    csv_path = p / "model_comparison.csv"
    json_path = p / "evaluation_summary.json"

    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(detailed_metrics, fp, indent=2)

    return csv_path, json_path
