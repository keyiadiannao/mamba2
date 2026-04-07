"""Build a balanced k-ary tree from leaf strings (bottom-up); embeddings are deterministic from text."""

from __future__ import annotations

import hashlib
import math
from typing import List

import torch

from src.rag_tree.tree import TreeNode


def _text_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1) + 1


def text_embedding(
    text: str,
    chunk_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic [chunk_len, dim] Gaussian-ish features from UTF-8 text (not a learned encoder)."""
    g = torch.Generator(device=device)
    g.manual_seed(_text_seed(text))
    return torch.randn(chunk_len, dim, device=device, dtype=dtype, generator=g)


def _required_depth(num_leaves: int, fanout: int) -> int:
    d = math.log(num_leaves) / math.log(fanout)
    if abs(d - round(d)) > 1e-9:
        raise ValueError(f"num_leaves={num_leaves} must be fanout**depth for fanout={fanout}")
    return int(round(d))


def build_bottom_up_text_tree(
    leaf_texts: List[str],
    fanout: int,
    chunk_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> TreeNode:
    """
    Leaves listed left-to-right; parents concatenate child texts with newlines.
    Requires len(leaf_texts) == fanout**depth for integer depth.
    """
    n = len(leaf_texts)
    if n < 1:
        raise ValueError("need at least one leaf text")
    _required_depth(n, fanout)

    current: List[TreeNode] = [
        TreeNode(
            embedding=text_embedding(t, chunk_len, dim, device, dtype),
            children=[],
            text=t,
        )
        for t in leaf_texts
    ]

    while len(current) > 1:
        nxt: List[TreeNode] = []
        for i in range(0, len(current), fanout):
            group = current[i : i + fanout]
            parent_text = "\n".join(node.text for node in group)
            emb = text_embedding(parent_text, chunk_len, dim, device, dtype)
            parent = TreeNode(embedding=emb, children=list(group), text=parent_text)
            nxt.append(parent)
        current = nxt

    return current[0]
