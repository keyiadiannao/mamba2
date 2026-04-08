"""
State-Snapshot Guided Search (SSGS) — 最小草稿。

不依赖 mamba-ssm：

- ``dfs_ssgs``：用路径上节点 id 列表模拟离散状态。
- ``TensorNavState`` + ``dfs_ssgs_tensor``：用 **真实 torch 向量 h** 做快照/恢复；
  每读一节点将 ``embedding`` 沿序列维均值累加到 h（占位式「一步更新」），便于测 **clone / copy_** 成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import torch

from .tree import TreeNode


State = List[Any]  # e.g. sequence of node ids along current path from root


@dataclass
class SSGSTrace:
    """可测事件序列（单测与日后 profiling 钩子）。"""

    events: List[str] = field(default_factory=list)
    snapshots_taken: int = 0
    rollbacks: int = 0
    leaf_checks: int = 0


def _node_key(node: TreeNode, key_fn: Callable[[TreeNode], Any]) -> Any:
    return key_fn(node)


def dfs_ssgs(
    node: TreeNode,
    prefix: State,
    *,
    key_fn: Callable[[TreeNode], Any] = id,
    leaf_goal: Callable[[TreeNode], bool],
    trace: Optional[SSGSTrace] = None,
    mount_snapshot: Optional[Callable[[TreeNode, State], None]] = None,
) -> Tuple[bool, State]:
    """
    深度优先：读完 ``node`` 后状态为 ``prefix + [key(node)]``。
    内部节点：在尝试每个子节点前打快照；子树全失败则返回 (False, checkpoint)。

    mount_snapshot(node, checkpoint) 可选，用于把快照挂到 ``TreeNode.state_snapshot`` 等。
    """
    t = trace
    cur = prefix + [_node_key(node, key_fn)]
    if t:
        t.events.append(f"visit:{_node_key(node, key_fn)}")

    if node.is_leaf():
        if t:
            t.leaf_checks += 1
        ok = leaf_goal(node)
        if t:
            t.events.append(f"leaf_ok:{ok}")
        return ok, cur

    checkpoint = cur.copy()
    if mount_snapshot is not None:
        mount_snapshot(node, checkpoint)
    if t:
        t.snapshots_taken += 1

    for ch in node.children:
        if t:
            t.events.append(f"try_child:{_node_key(ch, key_fn)}")
        ok, st = dfs_ssgs(ch, checkpoint, key_fn=key_fn, leaf_goal=leaf_goal, trace=t, mount_snapshot=mount_snapshot)
        if ok:
            return True, st
        if t:
            t.rollbacks += 1
            t.events.append(f"rollback->{_node_key(node, key_fn)}")

    return False, checkpoint


def mount_snapshot_on_tree(node: TreeNode, checkpoint: State) -> None:
    """把当前检查点（浅拷贝列表）挂到节点，供调试 / 论文图示。"""
    node.state_snapshot = list(checkpoint)


@dataclass
class TensorNavState:
    """
    固定形状隐向量 ``h``（[D]）。``absorb_node``：把节点 ``[L,D]`` embedding 的序列均值加到 h。
    与真实 SSM 不同，但 **snapshot=clone、restore=copy_** 与日后多层状态块同类。
    """

    h: torch.Tensor

    @classmethod
    def zeros(cls, dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TensorNavState:
        return cls(torch.zeros(dim, device=device, dtype=dtype))

    def absorb_node(self, node: TreeNode) -> None:
        self.h = self.h + node.embedding.mean(dim=0)

    def snapshot(self) -> torch.Tensor:
        return self.h.clone()

    def restore(self, snap: torch.Tensor) -> None:
        self.h.copy_(snap)


def mount_tensor_snapshot_on_tree(node: TreeNode, checkpoint: torch.Tensor) -> None:
    node.state_snapshot = checkpoint.detach().clone()


def dfs_ssgs_tensor(
    node: TreeNode,
    state: TensorNavState,
    *,
    leaf_goal: Callable[[TreeNode], bool],
    trace: Optional[SSGSTrace] = None,
    mount_snapshot: Optional[Callable[[TreeNode, torch.Tensor], None]] = None,
) -> bool:
    """
    与 ``dfs_ssgs`` 同序 DFS；内部节点在子树前 ``snapshot``，试兄弟前 ``restore``。
    返回是否找到满足 ``leaf_goal`` 的叶。
    """
    state.absorb_node(node)
    t = trace
    if t:
        t.events.append("tensor_visit")

    if node.is_leaf():
        if t:
            t.leaf_checks += 1
        ok = leaf_goal(node)
        if t:
            t.events.append(f"leaf_ok:{ok}")
        return ok

    checkpoint = state.snapshot()
    if mount_snapshot is not None:
        mount_snapshot(node, checkpoint)
    if t:
        t.snapshots_taken += 1

    for ch in node.children:
        state.restore(checkpoint)
        if t:
            t.events.append("tensor_try_child")
        if dfs_ssgs_tensor(ch, state, leaf_goal=leaf_goal, trace=t, mount_snapshot=mount_snapshot):
            return True
        if t:
            t.rollbacks += 1
            t.events.append("tensor_rollback")

    return False


def clear_tree_snapshots(root: TreeNode) -> None:
    """递归清空 ``state_snapshot``。"""

    def _walk(n: TreeNode) -> None:
        n.state_snapshot = None
        for c in n.children:
            _walk(c)

    _walk(root)
