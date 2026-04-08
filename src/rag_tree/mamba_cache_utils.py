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
