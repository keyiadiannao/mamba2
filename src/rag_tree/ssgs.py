"""
State-Snapshot Guided Search (SSGS) — 最小草稿。

- ``dfs_ssgs``：用路径上节点 id 列表模拟离散状态。
- ``TensorNavState`` + ``dfs_ssgs_tensor``：用 **真实 torch 向量 h** 做快照/恢复（占位更新）。
- ``MambaNavState`` + ``dfs_ssgs_mamba``：用 **HF ``Mamba2Model`` + ``DynamicCache``**；每节点按 **token**
  增量前向（``seq_len=1``）。**CUDA** 上 ``build_toy_mamba2_for_ssgs`` 会 **patch 每层 mixer 为 ``torch_forward``**，
  避免 fused ``causal_conv1d`` 在 **batch=1** 下的 stride=8 报错（与 path-batch fused 峰值 **不同**）。快照为
  ``conv_states``/``recurrent_states`` 的 **clone**，恢复为 **zero_ + copy_**（与 §7 S4 同语义）。

不依赖 ``mamba-ssm`` 亦可跑（HF naive）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import torch

from .mamba_cache_utils import (
    clone_mamba_dynamic_cache,
    patch_mamba2_model_use_torch_forward_only,
    restore_mamba_dynamic_cache,
    snapshot_list_nbytes,
    zero_mamba_dynamic_cache,
)
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


def build_toy_mamba2_for_ssgs(
    dim: int,
    device: torch.device,
    *,
    num_layers: int = 2,
) -> Any:
    """与 ``benchmark_mamba2_cache_snapshot_segments`` 默认宽度一致的小 ``Mamba2Model``（``use_cache=True``）。"""
    from transformers import Mamba2Config, Mamba2Model

    expand = 2
    inner = dim * expand
    head_dim = 32
    num_heads = inner // head_dim
    if inner % head_dim != 0:
        head_dim = 64
        num_heads = inner // head_dim
    cfg = Mamba2Config(
        num_hidden_layers=num_layers,
        hidden_size=dim,
        state_size=16,
        vocab_size=32000,
        num_heads=num_heads,
        head_dim=head_dim,
        expand=expand,
        n_groups=1,
        use_cache=True,
    )
    model = Mamba2Model(cfg).to(device)
    model.eval()
    if device.type == "cuda":
        patch_mamba2_model_use_torch_forward_only(model)
    return model


@dataclass
class MambaNavState:
    """
    导航态 = HF ``Mamba2Model`` 的 ``cache_params``（``DynamicCache``）。
    ``absorb_node``：对节点 ``embedding [L,D]`` 按 **单 token** 前向（与 ``has_previous_state`` 的增量解码一致）。
    """

    model: Any
    _cache: Any = None

    def absorb_node(self, node: TreeNode) -> None:
        dev = next(self.model.parameters()).device
        emb = node.embedding.unsqueeze(0).to(dev)
        with torch.no_grad():
            for ti in range(emb.shape[1]):
                tok = emb[:, ti : ti + 1, :]
                out = self.model(
                    inputs_embeds=tok,
                    cache_params=self._cache,
                    use_cache=True,
                    return_dict=True,
                )
                self._cache = out.cache_params

    def snapshot(self) -> Optional[List[dict[str, torch.Tensor]]]:
        if self._cache is None:
            return None
        return clone_mamba_dynamic_cache(self._cache)

    def restore(self, snap: Optional[List[dict[str, torch.Tensor]]]) -> None:
        if snap is None:
            self._cache = None
            return
        if self._cache is None:
            raise RuntimeError("MambaNavState.restore: cannot apply non-None snapshot when live cache is None")
        dev = next(self.model.parameters()).device
        zero_mamba_dynamic_cache(self._cache)
        restore_mamba_dynamic_cache(self._cache, snap, snapshot_on_cpu=False, device=dev)


def mount_mamba_cache_meta_on_tree(node: TreeNode, checkpoint: Optional[List[dict[str, torch.Tensor]]]) -> None:
    """将快照元数据挂到 ``node.state_snapshot``（避免整份张量挂在树上占内存）。"""
    if checkpoint is None:
        node.state_snapshot = None
    else:
        node.state_snapshot = {
            "kind": "mamba_dynamic_cache",
            "snapshot_nbytes": snapshot_list_nbytes(checkpoint),
        }


def dfs_ssgs_mamba(
    node: TreeNode,
    state: MambaNavState,
    *,
    leaf_goal: Callable[[TreeNode], bool],
    trace: Optional[SSGSTrace] = None,
    mount_snapshot: Optional[Callable[[TreeNode, Optional[List[dict[str, torch.Tensor]]]], None]] = None,
) -> bool:
    """
    与 ``dfs_ssgs_tensor`` 同序 DFS；内部节点在子树前 ``snapshot``，试兄弟前 ``restore``。
    要求 ``transformers`` 提供 ``Mamba2Model``。
    """
    state.absorb_node(node)
    t = trace
    if t:
        t.events.append("mamba_visit")

    if node.is_leaf():
        if t:
            t.leaf_checks += 1
        ok = leaf_goal(node)
        if t:
            t.events.append(f"leaf_ok:{ok}")
        return ok

    checkpoint = state.snapshot()
    if checkpoint is None:
        raise RuntimeError("dfs_ssgs_mamba: expected non-None cache after absorbing an internal node")
    if mount_snapshot is not None:
        mount_snapshot(node, checkpoint)
    if t:
        t.snapshots_taken += 1

    for ch in node.children:
        state.restore(checkpoint)
        if t:
            t.events.append("mamba_try_child")
        if dfs_ssgs_mamba(ch, state, leaf_goal=leaf_goal, trace=t, mount_snapshot=mount_snapshot):
            return True
        if t:
            t.rollbacks += 1
            t.events.append("mamba_rollback")

    return False


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
