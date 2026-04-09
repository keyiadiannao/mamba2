"""Path readers for micro-benchmarks: Transformer, GRU stand-in, and Mamba-2 (via Transformers)."""

from __future__ import annotations

import torch
import torch.nn as nn


def mamba2_path_reader_available() -> bool:
    try:
        from transformers import Mamba2Config, Mamba2Model  # noqa: F401

        return True
    except ImportError:
        return False


class TransformerPathReader(nn.Module):
    """Full-sequence self-attention over the concatenated path tokens.

    Complexity is **O(T²)** in path length **T** (and linear in batch of paths). This is **not** the
    incremental **TF-KV** trunk used in ``scripts/research/benchmark_tf_kv_path_segments.py`` (§7.2).
    """

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


def _mamba2_head_split(inner: int, head_dim: int | None) -> tuple[int, int]:
    """Return ``(num_heads, head_dim)`` for HF ``Mamba2Config`` (``inner = mamba_hidden * expand``).

    Fused ``causal_conv1d`` on CUDA often fails when ``num_heads < 8`` (stride/layout constraints);
    see ``SERVER_SWEEP_RUNBOOK.md`` and ``probe_mamba2_outputs.py``. When ``head_dim`` is ``None``,
    pick the **smallest** ``num_heads >= 8`` dividing ``inner`` so ``head_dim`` stays reasonably large.
    """
    if head_dim is not None:
        if inner % head_dim != 0:
            raise ValueError(f"mamba_hidden*expand ({inner}) must be divisible by head_dim ({head_dim})")
        return inner // head_dim, head_dim
    for nh in range(8, inner + 1):
        if inner % nh == 0:
            return nh, inner // nh
    for nh in range(7, 0, -1):
        if inner % nh == 0:
            return nh, inner // nh
    raise ValueError(f"cannot split inner={inner} into integer heads")


class Mamba2PathReader(nn.Module):
    """
    Mamba-2 stack on path embeddings [B, T, D] using `inputs_embeds` (no token embedding).

    Internal width `mamba_hidden` uses a small SSD-consistent config (expand=2, n_groups=1).
    Requires: pip install transformers (5.x for Mamba2).

    Default ``head_dim=None`` picks a fused-friendly **num_heads ≥ 8** split when possible.
    """

    def __init__(
        self,
        dim: int,
        *,
        mamba_hidden: int = 128,
        num_layers: int = 2,
        state_size: int = 16,
        vocab_size: int = 32000,
        head_dim: int | None = None,
        expand: int = 2,
    ) -> None:
        super().__init__()
        from transformers import Mamba2Config, Mamba2Model

        inner = mamba_hidden * expand
        num_heads, head_dim = _mamba2_head_split(inner, head_dim)
        self.mamba_hidden = mamba_hidden
        self.in_proj = nn.Linear(dim, mamba_hidden) if dim != mamba_hidden else nn.Identity()
        cfg = Mamba2Config(
            num_hidden_layers=num_layers,
            hidden_size=mamba_hidden,
            state_size=state_size,
            vocab_size=vocab_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand=expand,
            n_groups=1,
            use_cache=False,
        )
        self.core = Mamba2Model(cfg)
        self.out_proj = nn.Linear(mamba_hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if isinstance(self.in_proj, nn.Identity):
            h = x
        else:
            h = self.in_proj(x)
        h = h.contiguous()
        y = self.core(inputs_embeds=h).last_hidden_state
        pooled = y.mean(dim=1)
        return self.out_proj(pooled)
