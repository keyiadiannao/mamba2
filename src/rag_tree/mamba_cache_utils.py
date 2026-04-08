"""HF ``Mamba2Model`` ``DynamicCache``：按层 ``conv_states`` / ``recurrent_states`` 的 nbytes / clone / zero / restore。"""

from __future__ import annotations

import torch


def mamba_cache_tensor_nbytes(cache: object) -> int:
    total = 0
    layers = getattr(cache, "layers", None) or []
    for layer in layers:
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                total += int(t.numel() * t.element_size())
    return total


def clone_mamba_dynamic_cache(cache: object) -> list[dict[str, torch.Tensor]]:
    out: list[dict[str, torch.Tensor]] = []
    layers = getattr(cache, "layers", None) or []
    for layer in layers:
        block: dict[str, torch.Tensor] = {}
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                block[name] = t.clone().detach()
        out.append(block)
    return out


def zero_mamba_dynamic_cache(cache: object) -> None:
    layers = getattr(cache, "layers", None) or []
    for layer in layers:
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                t.zero_()


def restore_mamba_dynamic_cache(
    cache: object,
    snapshot: list[dict[str, torch.Tensor]],
    *,
    snapshot_on_cpu: bool = False,
    device: torch.device | None = None,
) -> None:
    layers = getattr(cache, "layers", None) or []
    dev = device or torch.device("cpu")
    for li, layer in enumerate(layers):
        for name in ("conv_states", "recurrent_states"):
            live = getattr(layer, name, None)
            if not torch.is_tensor(live):
                continue
            snap = snapshot[li][name]
            if snapshot_on_cpu:
                live.copy_(snap.to(dev, dtype=live.dtype, non_blocking=False))
            else:
                live.copy_(snap)


def snapshot_list_nbytes(snapshot: list[dict[str, torch.Tensor]]) -> int:
    n = 0
    for block in snapshot:
        for t in block.values():
            n += int(t.numel() * t.element_size())
    return n


def patch_mamba2_model_use_torch_forward_only(model: torch.nn.Module) -> None:
    """
    将每层 ``Mamba2Mixer.forward`` 改为始终调用 ``torch_forward``。

    **原因**：CUDA + fused ``causal_conv1d_fn`` 在 **batch=1**、token 步进（SSGS ``absorb_node``）下常触发
    ``strides ... multiples of 8``；``torch_forward`` 走 PyTorch ``conv1d`` / SSD 实现，语义与 cache 一致，仅更慢。
    CPU 上 HF 本身已走非 fused 路径，本函数可重复调用（幂等）。
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        return
    for block in layers:
        mixer = getattr(block, "mixer", None)
        if mixer is None or getattr(mixer, "_ssgs_torch_forward_patched", False):
            continue

        def _bind(m: torch.nn.Module):
            def forward(
                hidden_states: torch.Tensor,
                cache_params: object | None = None,
                attention_mask: torch.Tensor | None = None,
                **kwargs: object,
            ) -> torch.Tensor:
                return m.torch_forward(hidden_states, cache_params, attention_mask)

            return forward

        mixer.forward = _bind(mixer)  # type: ignore[method-assign]
        mixer._ssgs_torch_forward_patched = True
