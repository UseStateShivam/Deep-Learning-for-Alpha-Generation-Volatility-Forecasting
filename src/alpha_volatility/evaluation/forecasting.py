"""
Forecasting Evaluation Metrics
Computes MAE, RMSE, QLIKE, directional accuracy, and Student-t NLL.
"""

from typing import Dict, Optional

import numpy as np
import scipy.stats as stats


def compute_forecasting_metrics(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    y_true_vol: np.ndarray,
    y_pred_vol: np.ndarray,
    y_pred_scale: Optional[np.ndarray] = None,
    y_pred_df: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes standard point forecast and probabilistic metrics for returns and volatility.

    Args:
        y_true_ret: Realized forward log returns.
        y_pred_ret: Predicted expected returns.
        y_true_vol: Realized forward volatility.
        y_pred_vol: Predicted forward volatility.
        y_pred_scale: Predicted return scale parameter (optional).
        y_pred_df: Predicted return degrees of freedom (optional).

    Returns:
        Dictionary of scalar metrics.
    """
    valid_ret = ~np.isnan(y_true_ret) & ~np.isnan(y_pred_ret)
    valid_vol = ~np.isnan(y_true_vol) & ~np.isnan(y_pred_vol)

    metrics: Dict[str, float] = {}

    if np.sum(valid_ret) > 0:
        yt_r = y_true_ret[valid_ret]
        yp_r = y_pred_ret[valid_ret]

        # 1. Return Metrics
        ret_err = yp_r - yt_r
        metrics["return_mae"] = float(np.mean(np.abs(ret_err)))
        metrics["return_rmse"] = float(np.sqrt(np.mean(ret_err ** 2)))

        # Directional accuracy
        dir_correct = (np.sign(yp_r) == np.sign(yt_r)).astype(float)
        metrics["return_directional_accuracy"] = float(np.mean(dir_correct))

        # Time-series correlation
        if len(yt_r) > 2 and np.std(yp_r) > 1e-8 and np.std(yt_r) > 1e-8:
            metrics["return_pearson_corr"] = float(np.corrcoef(yp_r, yt_r)[0, 1])
        else:
            metrics["return_pearson_corr"] = 0.0

        # Distributional NLL if scale and df are supplied
        if y_pred_scale is not None and y_pred_df is not None:
            scale_sub = y_pred_scale[valid_ret]
            df_sub = y_pred_df[valid_ret]
            # Student-t logpdf
            log_liks = stats.t.logpdf(yt_r, df=df_sub, loc=yp_r, scale=np.maximum(scale_sub, 1e-6))
            metrics["return_nll"] = float(-np.mean(log_liks))

    if np.sum(valid_vol) > 0:
        yt_v = np.maximum(y_true_vol[valid_vol], 1e-6)
        yp_v = np.maximum(y_pred_vol[valid_vol], 1e-6)

        vol_err = yp_v - yt_v
        metrics["volatility_mae"] = float(np.mean(np.abs(vol_err)))
        metrics["volatility_rmse"] = float(np.sqrt(np.mean(vol_err ** 2)))

        # QLIKE: y / y_hat - ln(y / y_hat) - 1
        ratio = yt_v / yp_v
        qlike = ratio - np.log(ratio) - 1.0
        metrics["volatility_qlike"] = float(np.mean(qlike))

    return metrics
