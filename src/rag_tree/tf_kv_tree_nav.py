"""
TF-KV toy trunk on the same DFS order as ``dfs_ssgs_mamba``: internal-node snapshot, sibling restore.

``absorb_node`` feeds one tree node as a single ``forward_chunk`` (``chunk_len`` tokens), with
``pos_offset`` = cumulative tokens on the current path prefix — matching Mamba's per-node sequence
length in the tree reader convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from .tf_kv_incremental import IncrementalCausalTransformerKV
from .tree import TreeNode

# Re-use trace type from ssgs for uniform JSON counters.
from .ssgs import SSGSTrace


def _clone_kv_past(past: list[tuple[torch.Tensor, torch.Tensor] | None]) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    out: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for p in past:
        if p is None:
            out.append(None)
        else:
            K, V = p
            out.append((K.detach().clone(), V.detach().clone()))
    return out


def _apply_kv_past(model: IncrementalCausalTransformerKV, state: list[tuple[torch.Tensor, torch.Tensor] | None]) -> None:
    model._past = []
    for p in state:
        if p is None:
            model._past.append(None)
        else:
            K, V = p
            model._past.append((K.clone(), V.clone()))


@dataclass
class TfKvNavState:
    model: IncrementalCausalTransformerKV
    token_count: int = 0

    def reset(self) -> None:
        self.model.reset()
        self.token_count = 0

    def absorb_node(self, node: TreeNode) -> None:
        x = node.embedding.unsqueeze(0)
        with torch.no_grad():
            self.model.forward_chunk(x, pos_offset=self.token_count)
        self.token_count += int(x.shape[1])

    def snapshot(self) -> tuple[list[tuple[torch.Tensor, torch.Tensor] | None], int]:
        return (_clone_kv_past(self.model._past), self.token_count)

    def restore(self, snap: tuple[list[tuple[torch.Tensor, torch.Tensor] | None], int]) -> None:
        past, tc = snap
        _apply_kv_past(self.model, past)
        self.token_count = tc


def mount_tf_kv_meta_on_tree(node: TreeNode, state: TfKvNavState) -> None:
    node.state_snapshot = {
        "kind": "tf_kv_prefix",
        "kv_nbytes": int(state.model.kv_nbytes()),
        "token_prefix_tokens": int(state.token_count),
    }


@dataclass
class TfKvTruncateNavState:
    """
    Same DFS order as ``TfKvNavState``, but checkpoint is only **prefix length** (tokens).
    ``restore`` = ``truncate_kv(keep_tokens)`` + reset ``token_count`` — aligns with §7.2
    ``--branch-truncate-demo`` semantics (drop wrong-branch suffix, keep fork prefix).
    """

    model: IncrementalCausalTransformerKV
    token_count: int = 0
    truncate_kv_calls: int = 0

    def reset(self) -> None:
        self.model.reset()
        self.token_count = 0
        self.truncate_kv_calls = 0

    def absorb_node(self, node: TreeNode) -> None:
        x = node.embedding.unsqueeze(0)
        with torch.no_grad():
            self.model.forward_chunk(x, pos_offset=self.token_count)
        self.token_count += int(x.shape[1])

    def snapshot(self) -> int:
        return self.token_count

    def restore(self, snap: int) -> None:
        self.model.truncate_kv(snap)
        self.token_count = snap
        self.truncate_kv_calls += 1


def mount_tf_kv_truncate_meta_on_tree(node: TreeNode, state: TfKvTruncateNavState) -> None:
    node.state_snapshot = {
        "kind": "tf_kv_prefix_truncate",
        "kv_nbytes": int(state.model.kv_nbytes()),
        "token_prefix_tokens": int(state.token_count),
    }


TfKvNavStateUnion = Union[TfKvNavState, TfKvTruncateNavState]


def dfs_tf_kv_nav(
    node: TreeNode,
    state: TfKvNavStateUnion,
    *,
    leaf_goal: Callable[[TreeNode], bool],
    trace: Optional[SSGSTrace] = None,
    mount_snapshot: Optional[Callable[[TreeNode, TfKvNavStateUnion], None]] = None,
) -> bool:
    state.absorb_node(node)
    t = trace
    if t:
        t.events.append("tf_kv_visit")

    if node.is_leaf():
        if t:
            t.leaf_checks += 1
        ok = leaf_goal(node)
        if t:
            t.events.append(f"leaf_ok:{ok}")
        return ok

    checkpoint = state.snapshot()
    if mount_snapshot is not None:
        mount_snapshot(node, state)
    if t:
        t.snapshots_taken += 1

    for ch in node.children:
        state.restore(checkpoint)
        if t:
            t.events.append("tf_kv_try_child")
        if dfs_tf_kv_nav(ch, state, leaf_goal=leaf_goal, trace=t, mount_snapshot=mount_snapshot):
            return True
        if t:
            t.rollbacks += 1
            t.events.append("tf_kv_rollback")

    return False
