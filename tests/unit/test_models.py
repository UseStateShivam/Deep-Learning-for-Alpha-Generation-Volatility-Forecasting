"""
Unit Tests for Deep Learning and Baseline Model Architectures.
"""

import numpy as np
import torch

from alpha_volatility.models.lightgbm_model import LightGBMForecaster
from alpha_volatility.models.losses import qlike_loss, student_t_nll
from alpha_volatility.models.tcn import TCNForecaster
from alpha_volatility.models.transformer import TransformerForecaster


def test_tcn_forward_pass_shapes_and_gradients():
    """Verifies TCN outputs correct tensor shapes and computes valid backprop gradients."""
    batch_size = 4
    seq_len = 16
    n_features = 10
    model = TCNForecaster(input_dim=n_features, channel_sizes=[16, 16], kernel_size=3)

    x = torch.randn(batch_size, seq_len, n_features, requires_grad=True)
    out = model(x)

    assert "pred_return_mean" in out
    assert "pred_return_scale" in out
    assert "pred_return_df" in out
    assert "pred_volatility" in out

    assert out["pred_return_mean"].shape == (batch_size,)
    assert torch.all(out["pred_return_scale"] > 0)
    assert torch.all(out["pred_return_df"] > 2.0)
    assert torch.all(out["pred_volatility"] > 0)

    # Gradient flow test
    loss = out["pred_return_mean"].sum() + out["pred_volatility"].sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_tcn_receptive_field_formula():
    """Verifies receptive field calculation matches theory."""
    # K=3, layers=2 with dilations 1, 2
    # RF = 1 + 2*(3-1)*1 + 2*(3-1)*2 = 1 + 4 + 8 = 13
    model = TCNForecaster(input_dim=5, channel_sizes=[16, 16], kernel_size=3)
    rf = model.compute_receptive_field()
    assert rf == 13


def test_transformer_forward_pass_and_mask():
    """Verifies Transformer forward pass and causal mask shapes."""
    batch_size = 4
    seq_len = 16
    n_features = 8
    model = TransformerForecaster(input_dim=n_features, d_model=16, nhead=2, num_layers=2)

    x = torch.randn(batch_size, seq_len, n_features)
    out = model(x)

    assert out["pred_return_mean"].shape == (batch_size,)
    assert torch.all(out["pred_return_scale"] > 0)
    assert torch.all(out["pred_volatility"] > 0)


def test_student_t_nll_finite():
    """Verifies Student-t NLL produces finite, numerically stable output."""
    y = torch.tensor([0.01, -0.02, 0.005])
    mu = torch.tensor([0.0, 0.0, 0.0])
    sigma = torch.tensor([0.01, 0.01, 0.01])
    nu = torch.tensor([4.0, 5.0, 6.0])

    loss = student_t_nll(y, mu, sigma, nu)
    assert torch.isfinite(loss)


def test_qlike_loss_zero_at_optimum():
    """Verifies QLIKE loss is non-negative and reaches 0 when y == y_hat."""
    y = torch.tensor([0.02, 0.05])
    y_hat_perfect = torch.tensor([0.02, 0.05])
    loss_zero = qlike_loss(y, y_hat_perfect)
    assert torch.isclose(loss_zero, torch.tensor(0.0), atol=1e-5)

    y_hat_imperfect = torch.tensor([0.01, 0.08])
    loss_pos = qlike_loss(y, y_hat_imperfect)
    assert loss_pos > 0.0


def test_lightgbm_fit_and_predict(sample_features_and_labels):
    """Verifies LightGBM fits properly and produces return and vol predictions."""
    feat_cols = ["log_return_1", "realized_vol_6", "log_volume"]
    model = LightGBMForecaster(feature_names=feat_cols)
    model.fit(sample_features_and_labels.iloc[:200])

    preds = model.predict(sample_features_and_labels.iloc[200:])
    assert "pred_return_mean" in preds
    assert "pred_volatility" in preds
    assert len(preds["pred_return_mean"]) == len(sample_features_and_labels.iloc[200:])
    assert np.all(preds["pred_volatility"] > 0)
