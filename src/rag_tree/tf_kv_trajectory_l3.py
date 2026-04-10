"""
**L3 最小轨迹对照**（``RESEARCH_STATUS_AND_DIRECTION.md`` §3.5）：玩具 **TF-KV** 上硬编码读序。

**轨迹乙**：仅沿 **金路径**（根 → 目标叶）逐节点 ``forward_chunk``。  
**轨迹甲**：金路径前缀 → **快照** → 走入 **一条错枝**（根下与金路径不同的第一子节点，吸收该子树根）→ **restore** → 再沿金路径剩余节点走到目标叶。

在 **full KV clone** 与 **truncate_kv** 两种恢复语义下，比较甲/乙 **末 token trunk hidden** 的余弦；**≈1** 表示「错探后回到分叉点再前进」与「直达」的表示一致。

**与 path-batch / M1 DFS** 不同 **kind**；禁止无脚注与主表合并。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from .tf_kv_incremental import IncrementalCausalTransformerKV
from .tf_kv_tree_nav import TfKvNavState, TfKvTruncateNavState
from .tree import (
    TreeNode,
    build_balanced_tree,
    find_root_leaf_path_ending_at,
    iter_root_leaf_paths,
)


def wrong_sibling_first_on_path(root: TreeNode, gold_path: List[TreeNode]) -> TreeNode:
    """First child of ``root`` that is not ``gold_path[1]`` (requires fanout >= 2)."""
    if len(gold_path) < 2:
        raise ValueError("gold_path must have root and at least one child")
    if gold_path[0] is not root:
        raise ValueError("gold_path[0] must be root")
    gold_child = gold_path[1]
    for ch in root.children:
        if ch is not gold_child:
            return ch
    raise ValueError("no alternate child at fork (fanout < 2?)")


def _cos_l2(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af, bf = a.float(), b.float()
    cos_t = F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0), dim=-1)
    cos = float(cos_t.squeeze().item())
    l2 = float((af - bf).norm().item())
    return cos, l2


def run_trajectory_b_direct_gold(
    model: IncrementalCausalTransformerKV,
    gold_path: List[TreeNode],
) -> torch.Tensor:
    model.reset()
    st = TfKvNavState(model=model)
    for node in gold_path:
        st.absorb_node(node)
    return model.read_last_token_hidden()


def run_trajectory_a_wrong_restore(
    model: IncrementalCausalTransformerKV,
    gold_path: List[TreeNode],
    *,
    use_truncate_restore: bool,
) -> torch.Tensor:
    root = gold_path[0]
    wrong_root_child = wrong_sibling_first_on_path(root, gold_path)
    model.reset()
    if use_truncate_restore:
        st: TfKvNavState | TfKvTruncateNavState = TfKvTruncateNavState(model=model)
    else:
        st = TfKvNavState(model=model)

    st.absorb_node(root)
    snap = st.snapshot()
    st.absorb_node(wrong_root_child)
    st.restore(snap)
    for node in gold_path[1:]:
        st.absorb_node(node)
    return model.read_last_token_hidden()


@dataclass(frozen=True)
class TrajectoryL3Result:
    cosine_ab: float
    l2_ab: float
    gold_path_nodes: int
    wrong_branch_node_is_leaf: bool
    wall_s_b: float
    wall_s_a: float


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def compare_trajectories_ab(
    *,
    depth: int,
    fanout: int,
    chunk_len: int,
    dim: int,
    tf_layers: int,
    nhead: int,
    ff_mult: int,
    device: torch.device,
    init_seed: int,
    use_truncate_restore: bool,
) -> tuple[dict[str, object], TrajectoryL3Result]:
    g = torch.Generator(device=device)
    g.manual_seed(init_seed)

    root = build_balanced_tree(depth, fanout, chunk_len, dim, device, generator=g)
    paths = list(iter_root_leaf_paths(root))
    if not paths:
        raise ValueError("empty tree")
    target_leaf = paths[-1][-1]
    gold_path = find_root_leaf_path_ending_at(root, target_leaf)
    if fanout < 2 or len(root.children) < 2:
        raise ValueError("trajectory L3 minimal requires fanout >= 2")

    wrong_node = wrong_sibling_first_on_path(root, gold_path)

    model = IncrementalCausalTransformerKV(
        dim=dim, nhead=nhead, num_layers=tf_layers, ff_mult=ff_mult
    ).to(device)
    model.eval()

    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        h_b = run_trajectory_b_direct_gold(model, gold_path)
        _sync(device)
        wall_b = time.perf_counter() - t0

        sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(sd0)
        _sync(device)
        t1 = time.perf_counter()
        h_a = run_trajectory_a_wrong_restore(
            model, gold_path, use_truncate_restore=use_truncate_restore
        )
        _sync(device)
        wall_a = time.perf_counter() - t1

    cos_ab, l2_ab = _cos_l2(h_a, h_b)
    res = TrajectoryL3Result(
        cosine_ab=round(cos_ab, 8),
        l2_ab=round(l2_ab, 8),
        gold_path_nodes=len(gold_path),
        wrong_branch_node_is_leaf=wrong_node.is_leaf(),
        wall_s_b=round(wall_b, 8),
        wall_s_a=round(wall_a, 8),
    )

    arm = "truncate_kv_restore" if use_truncate_restore else "full_kv_clone_restore"
    payload: dict[str, object] = {
        "arm": arm,
        "cosine_hidden_a_vs_b": res.cosine_ab,
        "l2_diff_hidden_a_vs_b": res.l2_ab,
        "gold_path_nodes": res.gold_path_nodes,
        "wrong_branch_first_step_is_leaf": res.wrong_branch_node_is_leaf,
        "wall_s_trajectory_b_direct": res.wall_s_b,
        "wall_s_trajectory_a_wrong_restore": res.wall_s_a,
        "definition": (
            "Trajectory B: forward only along gold root—target path. "
            "Trajectory A: gold prefix (root only) → snapshot → absorb wrong sibling subtree root "
            "→ restore → gold suffix. Compare last-token hidden of IncrementalCausalTransformerKV."
        ),
    }
    return payload, res
