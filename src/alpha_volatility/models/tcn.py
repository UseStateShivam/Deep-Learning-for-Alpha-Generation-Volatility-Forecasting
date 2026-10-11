"""
Temporal Convolutional Network (TCN) Architecture
Implements causal, dilated residual convolutions for sequence modeling in financial time series.
Guarantees strict causality: output at time t depends ONLY on inputs at tau <= t.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    Trims the right side of the convolution output to enforce strict causality.
    When left-padding of size P = (kernel_size - 1) * dilation is added before standard Conv1d,
    the output length is L + P. Chomp1d removes the trailing P elements so output length is L.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual block with two causal dilated Conv1d layers, weight norm, ReLU, and dropout.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Multi-layer dilated Temporal Convolutional Network.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Channels, SeqLen]
        return self.network(x)


class TCNForecaster(nn.Module):
    """
    End-to-end multi-task TCN forecasting model for:
    1. Student-t Return Distribution parameters (mu, sigma, nu)
    2. Realized Volatility forecast (strictly positive via Softplus)
    """

    def __init__(
        self,
        input_dim: int,
        channel_sizes: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        min_df: float = 2.1,
        max_df: float = 30.0,
    ):
        if channel_sizes is None:
            channel_sizes = [64, 64, 64]
        super().__init__()
        self.input_dim = input_dim
        self.channel_sizes = channel_sizes
        self.kernel_size = kernel_size
        self.min_df = min_df
        self.max_df = max_df

        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=channel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        final_channels = channel_sizes[-1]

        # Multi-task heads applied to the final sequence timestep representation
        # 1. Return Head: Location, Scale, Degrees of Freedom
        self.head_mu = nn.Linear(final_channels, 1)
        self.head_sigma = nn.Linear(final_channels, 1)
        self.head_nu = nn.Linear(final_channels, 1)

        # 2. Volatility Head: Realized volatility point estimate
        self.head_vol = nn.Linear(final_channels, 1)

    def compute_receptive_field(self) -> int:
        """
        Calculates the theoretical receptive field in number of bars:
          RF = 1 + sum_{l=0}^{L-1} 2 * (K - 1) * 2^l
        """
        rf = 1
        for i in range(len(self.channel_sizes)):
            rf += 2 * (self.kernel_size - 1) * (2 ** i)
        return rf

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor [Batch, SeqLen, InputDim]

        Returns:
            Dictionary containing:
              - 'pred_return_mean': E[r] [B]
              - 'pred_return_scale': sigma [B]
              - 'pred_return_df': nu [B]
              - 'pred_volatility': sigma_vol [B]
        """
        # Transpose from [Batch, SeqLen, InputDim] to [Batch, InputDim, SeqLen]
        x_trans = x.transpose(1, 2)
        features = self.tcn(x_trans)  # [Batch, FinalChannels, SeqLen]

        # Extract latest timestep (t) representation for prediction
        rep = features[:, :, -1]  # [Batch, FinalChannels]

        # Return parameters
        mu = self.head_mu(rep).squeeze(-1)
        sigma = F.softplus(self.head_sigma(rep)).squeeze(-1) + 1e-5
        nu_raw = F.softplus(self.head_nu(rep)).squeeze(-1)
        nu = torch.clamp(self.min_df + nu_raw, max=self.max_df)

        # Volatility prediction (strictly positive)
        vol = F.softplus(self.head_vol(rep)).squeeze(-1) + 1e-6

        return {
            "pred_return_mean": mu,
            "pred_return_scale": sigma,
            "pred_return_df": nu,
            "pred_volatility": vol,
        }
