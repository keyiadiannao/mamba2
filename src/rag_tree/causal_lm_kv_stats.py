"""Causal LM forward + cache / KV 体积统计（工程 Sprint 2；与 path-batch reader 分列）."""

from __future__ import annotations

from typing import Any, Optional


def tensor_tree_nbytes(obj: Any) -> int:
    """Sum ``numel * element_size`` for nested tuples/lists; tensor-like 用 ``numel``/``element_size`` duck-typing（**不**强依赖 ``torch`` import）。"""
    if obj is None:
        return 0
    # HF DynamicCache / EncoderDecoderCache: must sum key + value (do not return after key_cache only).
    if hasattr(obj, "key_cache") or hasattr(obj, "value_cache"):
        kc = getattr(obj, "key_cache", None) or []
        vc = getattr(obj, "value_cache", None) or []
        return tensor_tree_nbytes(kc) + tensor_tree_nbytes(vc)
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_nbytes(x) for x in obj)
    numel = getattr(obj, "numel", None)
    el = getattr(obj, "element_size", None)
    if callable(numel) and callable(el):
        try:
            return int(numel() * el())
        except Exception:
            return 0
    # Mamba-style cache params — sum known tensor containers
    if hasattr(obj, "conv_states") or hasattr(obj, "ssm_states"):
        cs = getattr(obj, "conv_states", None) or []
        ss = getattr(obj, "ssm_states", None) or []
        return tensor_tree_nbytes(cs) + tensor_tree_nbytes(ss)
    return 0


def cache_nbytes_from_outputs(outputs: Any) -> int:
    """Best-effort cache size from HF ``CausalLMOutput*``."""
    pv = getattr(outputs, "past_key_values", None)
    if pv is not None:
        if hasattr(pv, "to_legacy_cache"):
            try:
                legacy = pv.to_legacy_cache()
                n = tensor_tree_nbytes(legacy)
                if n > 0:
                    return n
            except Exception:
                pass
        n = tensor_tree_nbytes(pv)
        if n > 0:
            return n
    cp = getattr(outputs, "cache_params", None)
    if cp is not None:
        return tensor_tree_nbytes(cp)
    return 0


def estimate_attention_kv_nbytes_mha_stack(
    config: Any,
    seq_len: int,
    batch: int = 1,
    element_size: int = 2,
) -> Optional[int]:
    """Rough nbytes of **attention** K+V across all layers (standard MHA: K,V each ``B×S×H`` per layer).

    Used when ``past_key_values`` is missing/empty on a full-sequence forward (some HF versions omit it).
    Not applicable to pure SSM / Mamba (no attention KV); returns ``None`` if config lacks layer/hidden fields.
    """
    mt = getattr(config, "model_type", None)
    if mt in ("mamba", "mamba2"):
        return None
    n_layer = getattr(config, "n_layer", None)
    if n_layer is None:
        n_layer = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "n_embd", None)
    if hidden is None:
        hidden = getattr(config, "hidden_size", None)
    if n_layer is None or hidden is None:
        return None
    # Per layer: K and V, each numel = B * S * hidden (aggregate over heads)
    return int(n_layer * 2 * batch * seq_len * hidden * element_size)
