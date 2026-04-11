"""No torch import — duck-typed tensor-like."""
from __future__ import annotations

from src.rag_tree.causal_lm_kv_stats import (
    cache_nbytes_from_outputs,
    estimate_attention_kv_nbytes_mha_stack,
    tensor_tree_nbytes,
)


class _FakeTensor:
    def __init__(self, n: int, es: int = 4) -> None:
        self._n = n
        self._es = es

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return self._es


def test_tensor_tree_nbytes_nested() -> None:
    a = _FakeTensor(6)
    b = (a, _FakeTensor(4))
    assert tensor_tree_nbytes(b) == 6 * 4 + 4 * 4


def test_tensor_tree_nbytes_none() -> None:
    assert tensor_tree_nbytes(None) == 0


def test_tensor_tree_nbytes_dynamic_cache_sums_key_and_value() -> None:
    """Empty key_cache must not hide nonempty value_cache (HF DynamicCache pattern)."""
    kc: list = []
    vc = [_FakeTensor(5)]
    obj = type("_C", (), {"key_cache": kc, "value_cache": vc})()
    assert tensor_tree_nbytes(obj) == 5 * 4


def test_cache_nbytes_from_outputs_tuple_past() -> None:
    class _Out:
        past_key_values = ((_FakeTensor(2), _FakeTensor(3)),)

    assert cache_nbytes_from_outputs(_Out()) == 2 * 4 + 3 * 4


def test_cache_nbytes_from_outputs_to_legacy_cache() -> None:
    class _Cache:
        def to_legacy_cache(self):
            return ((_FakeTensor(1), _FakeTensor(1)),)

    class _Out:
        past_key_values = _Cache()

    assert cache_nbytes_from_outputs(_Out()) == 8


def test_estimate_attention_kv_nbytes_gpt2_shape() -> None:
    class _Cfg:
        n_layer = 12
        n_embd = 768

    # 12 * 2 * 512 * 768 * 2
    assert estimate_attention_kv_nbytes_mha_stack(_Cfg(), seq_len=512, batch=1, element_size=2) == 18874368


def test_estimate_attention_kv_none_for_mamba2() -> None:
    class _Cfg:
        model_type = "mamba2"
        num_hidden_layers = 48
        hidden_size = 1024

    assert estimate_attention_kv_nbytes_mha_stack(_Cfg(), seq_len=512, batch=1, element_size=2) is None
