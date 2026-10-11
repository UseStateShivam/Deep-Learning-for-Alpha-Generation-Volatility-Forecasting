"""
Temporal Transformer Architecture
Implements a causal multi-head self-attention Transformer encoder for financial sequences.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for temporal sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, SeqLen, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Temporal Transformer encoder with multi-task prediction heads for
    Student-t return distributions and realized volatility.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
        use_causal_mask: bool = True,
        min_df: float = 2.1,
        max_df: float = 30.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_causal_mask = use_causal_mask
        self.min_df = min_df
        self.max_df = max_df

        # Linear projection from input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-task heads applied to the latest sequence element representation
        self.head_mu = nn.Linear(d_model, 1)
        self.head_sigma = nn.Linear(d_model, 1)
        self.head_nu = nn.Linear(d_model, 1)
        self.head_vol = nn.Linear(d_model, 1)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates an upper-triangular causal attention mask.
        Positions with True / -inf cannot be attended to.
        """
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        return mask

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
        batch_size, seq_len, _ = x.shape
        proj = self.input_proj(x)  # [B, L, d_model]
        encoded = self.pos_encoder(proj)

        if self.use_causal_mask:
            mask = self.generate_causal_mask(seq_len, x.device)
            hidden = self.transformer_encoder(encoded, mask=mask, is_causal=True)
        else:
            hidden = self.transformer_encoder(encoded)

        # Extract latest timestep (t) representation
        rep = hidden[:, -1, :]  # [B, d_model]

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
