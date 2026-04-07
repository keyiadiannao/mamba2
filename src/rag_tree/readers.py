"""Path readers for micro-benchmarks: Transformer (global attention) vs GRU (fixed state, SSM stand-in)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerPathReader(nn.Module):
    """Full-sequence self-attention over the concatenated path tokens."""

    def __init__(self, dim: int, nhead: int, num_layers: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError("dim must be divisible by nhead")
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.encoder(x)
        out = h.mean(dim=1)
        return self.pool(out)


class GRUPathReader(nn.Module):
    """Recurrent encoding; use as cheap stand-in until Mamba-2 is wired in."""

    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        _, h_n = self.gru(x)
        last = h_n[-1]
        return self.out_proj(last)
