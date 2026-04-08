"""
State-Snapshot Guided Search (SSGS) — 最小草稿。

不依赖 mamba-ssm：用「路径上已读节点 id 列表」模拟隐状态；在内部节点分叉前打快照，
子树失败则回到快照再试兄弟分支。后续可把 list[int] 换成 clone 的多层 tensor 元组。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

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


def clear_tree_snapshots(root: TreeNode) -> None:
    """递归清空 ``state_snapshot``。"""

    def _walk(n: TreeNode) -> None:
        n.state_snapshot = None
        for c in n.children:
            _walk(c)

    _walk(root)
