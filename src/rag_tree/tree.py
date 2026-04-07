"""Toy tree for path-wise RAG / navigation benchmarks (synthetic node embeddings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List

import torch


@dataclass
class TreeNode:
    """A node holds a fixed embedding (placeholder for one retrieved chunk summary)."""

    embedding: torch.Tensor  # [L, D] sequence of token vectors for this node
    children: List["TreeNode"] = field(default_factory=list)
    text: str = ""  # optional source text for text-shaped / RAPTOR-style trees

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def build_balanced_tree(
    depth: int,
    fanout: int,
    chunk_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> TreeNode:
    """
    Perfect k-ary tree of given depth (depth=0 => single leaf root).

    depth: number of edges from root to any leaf; root at depth 0 has depth layers below
           such that leaves are `depth` hops away (here: root-only tree if depth=0).
    """
    g = generator
    emb = torch.randn(chunk_len, dim, device=device, dtype=dtype, generator=g)
    if depth == 0:
        return TreeNode(embedding=emb, children=[])

    children = [
        build_balanced_tree(depth - 1, fanout, chunk_len, dim, device, dtype, g)
        for _ in range(fanout)
    ]
    return TreeNode(embedding=emb, children=children)


def iter_root_leaf_paths(root: TreeNode) -> Iterator[List[TreeNode]]:
    if root.is_leaf():
        yield [root]
        return
    for ch in root.children:
        for path in iter_root_leaf_paths(ch):
            yield [root, *path]


def path_tensor(path: List[TreeNode]) -> torch.Tensor:
    """Concatenate node chunks along time: [sum(chunk_len), D]."""
    return torch.cat([n.embedding for n in path], dim=0)


def batched_paths(root: TreeNode) -> tuple[torch.Tensor, int]:
    """
    Stack all root-to-leaf paths into [B, T, D] where T is constant per leaf
    only if all leaves share depth (balanced tree).
    """
    paths = list(iter_root_leaf_paths(root))
    if not paths:
        raise ValueError("empty tree")
    tensors = [path_tensor(p) for p in paths]
    t0 = tensors[0].shape[0]
    if not all(t.shape[0] == t0 for t in tensors):
        raise ValueError("varying path lengths; pad before batching (not implemented)")
    d = tensors[0].shape[1]
    batch = torch.stack(tensors, dim=0)  # [B, T, D]
    return batch, len(paths)
