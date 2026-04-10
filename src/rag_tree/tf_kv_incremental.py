"""
Pre-LN causal Transformer trunk with per-layer MHA K/V cache (§7.2 TF-KV toy protocol).

Shared by ``benchmark_tf_kv_path_segments`` and tree-navigation baselines.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _CausalLayerCached(nn.Module):
    """Pre-LN causal self-attn + FFN; MHA K/V cached per token prefix for incremental chunks."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout_p = dropout

    def forward_chunk(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        pos_offset: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, Tc, D]; pos_offset = number of tokens before this chunk
        B, Tc, D = x.shape
        H, Dh = self.nhead, self.head_dim
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, Tc, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]
        if past_kv is None:
            K = k_new
            V = v_new
        else:
            Kp, Vp = past_kv
            K = torch.cat([Kp, k_new], dim=2)
            V = torch.cat([Vp, v_new], dim=2)
        scale = Dh**-0.5
        outs: list[torch.Tensor] = []
        for i in range(Tc):
            qi = q[:, :, i : i + 1, :]
            k_len = pos_offset + i + 1
            Ks = K[:, :, :k_len, :]
            Vs = V[:, :, :k_len, :]
            scores = (qi @ Ks.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            outs.append(attn @ Vs)
        o = torch.cat(outs, dim=2).transpose(1, 2).contiguous().reshape(B, Tc, D)
        x = x + self.out_proj(o)
        x = x + self.ff(self.norm2(x))
        return x, (K, V)


class IncrementalCausalTransformerKV(nn.Module):
    def __init__(self, dim: int, nhead: int, num_layers: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        dim_ff = dim * ff_mult
        self.layers = nn.ModuleList(
            [_CausalLayerCached(dim, nhead, dim_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.reset()

    def reset(self) -> None:
        self._past: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.layers)
        self._last_hidden: Optional[torch.Tensor] = None

    def kv_nbytes(self) -> int:
        total = 0
        for past in self._past:
            if past is None:
                continue
            K, V = past
            total += int(K.numel() * K.element_size() + V.numel() * V.element_size())
        return total

    def truncate_kv(self, keep_tokens: int) -> None:
        for li, past in enumerate(self._past):
            if past is None:
                continue
            K, V = past
            if keep_tokens > K.shape[2]:
                raise ValueError(f"truncate keep_tokens={keep_tokens} > cache len {K.shape[2]}")
            self._past[li] = (
                K[:, :, :keep_tokens, :].contiguous(),
                V[:, :, :keep_tokens, :].contiguous(),
            )

    def forward_chunk(self, x: torch.Tensor, pos_offset: int) -> torch.Tensor:
        for li, layer in enumerate(self.layers):
            x, kv = layer.forward_chunk(x, self._past[li], pos_offset)
            self._past[li] = kv
        self._last_hidden = x
        return x

    def read_last_token_hidden(self) -> torch.Tensor:
        """Last forward_chunk output, final token position: shape ``[D]`` (batch=1)."""
        if self._last_hidden is None:
            raise RuntimeError("read_last_token_hidden: no forward_chunk yet")
        t = self._last_hidden[:, -1, :].squeeze(0)
        return t.detach()
