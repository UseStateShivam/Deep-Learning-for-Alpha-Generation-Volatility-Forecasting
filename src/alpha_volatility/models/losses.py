"""
Loss Functions Module
Implements Student-t NLL, Gaussian NLL, QLIKE volatility loss, and multi-task loss objectives.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def student_t_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes Negative Log-Likelihood for conditional Student-t distribution:
      p(y | mu, sigma, nu) = [Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(pi*nu) * sigma)]
                            * [1 + (y - mu)^2 / (nu * sigma^2)]^(-(nu+1)/2)

    Args:
        y: Realized target values [B] or [B, 1].
        mu: Location parameter [B] or [B, 1].
        sigma: Strictly positive scale parameter [B] or [B, 1].
        nu: Degrees of freedom (nu > 2) [B] or [B, 1].
        eps: Small constant for numerical stability.

    Returns:
        Scalar average NLL loss.
    """
    sigma = torch.clamp(sigma, min=eps)
    nu = torch.clamp(nu, min=2.01, max=100.0)

    term1 = torch.lgamma(0.5 * (nu + 1.0))
    term2 = torch.lgamma(0.5 * nu)
    term3 = 0.5 * torch.log(math.pi * nu)
    term4 = torch.log(sigma)

    z = (y - mu) / sigma
    term5 = 0.5 * (nu + 1.0) * torch.log1p((z ** 2) / nu)

    log_lik = term1 - term2 - term3 - term4 - term5
    return -torch.mean(log_lik)


def gaussian_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes Negative Log-Likelihood for conditional Gaussian distribution.
    """
    sigma = torch.clamp(sigma, min=eps)
    variance = sigma ** 2
    log_lik = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * ((y - mu) ** 2) / variance
    return -torch.mean(log_lik)


def qlike_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Quasi-Likelihood (QLIKE) volatility forecast loss:
      QLIKE(y, y_hat) = y / y_hat - ln(y / y_hat) - 1

    Properties:
      - Strictly >= 0 with minimum at y = y_hat.
      - Asymmetric penalty punishing underforecasting of risk more severely.

    Args:
        y_true: Realized volatility [B] or [B, 1].
        y_pred: Predicted volatility [B] or [B, 1] (strictly positive).
        eps: Lower clamp bound.

    Returns:
        Scalar mean QLIKE loss.
    """
    y = torch.clamp(y_true, min=eps)
    y_hat = torch.clamp(y_pred, min=eps)

    ratio = y / y_hat
    loss = ratio - torch.log(ratio) - 1.0
    return torch.mean(loss)


class MultiTaskLoss(nn.Module):
    """
    Combines Return Distribution NLL and Realized Volatility QLIKE/MSE.
    L = w_ret * L_ret + w_vol * L_vol
    """

    def __init__(
        self,
        return_distribution: str = "student_t",
        weight_return: float = 1.0,
        weight_volatility: float = 0.5,
        volatility_loss_type: str = "qlike",
    ):
        super().__init__()
        self.return_distribution = return_distribution
        self.weight_return = weight_return
        self.weight_volatility = weight_volatility
        self.volatility_loss_type = volatility_loss_type

    def forward(
        self,
        y_ret: torch.Tensor,
        y_vol: torch.Tensor,
        pred_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates weighted composite loss and returns breakdown metrics.
        """
        # 1. Return Task Loss
        mu = pred_dict["pred_return_mean"]
        sigma = pred_dict["pred_return_scale"]
        if self.return_distribution == "student_t":
            nu = pred_dict["pred_return_df"]
            l_ret = student_t_nll(y_ret, mu, sigma, nu)
        else:
            l_ret = gaussian_nll(y_ret, mu, sigma)

        # 2. Volatility Task Loss
        pred_vol = pred_dict["pred_volatility"]
        if self.volatility_loss_type == "qlike":
            l_vol = qlike_loss(y_vol, pred_vol)
        else:
            l_vol = F.mse_loss(pred_vol, y_vol)

        total_loss = self.weight_return * l_ret + self.weight_volatility * l_vol

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_return": float(l_ret.detach().item()),
            "loss_volatility": float(l_vol.detach().item()),
        }
        return total_loss, metrics
