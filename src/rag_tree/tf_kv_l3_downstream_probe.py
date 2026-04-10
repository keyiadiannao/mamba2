"""
**L3 downstream (minimal)**: fixed **leaf classification head** on last-token trunk hidden.

After ``dfs_tf_kv_nav`` reaches the target leaf, compare **cross-entropy** (fixed ``Linear(dim,
num_leaves)`` init, same ``target_leaf_index``) using **nav hidden** vs **gold-path-only** hidden
from a fresh same-weight trunk. When L3 hidden consistency holds, **``ce_nav`` ≈ ``ce_ref``** and
**``abs_ce_delta``** is near zero.

This is **not** the tree-LM **CE routing vs learned child head** harness (see **X-20260423** /
**X-20260424** in ``EXPERIMENT_REGISTRY``): there a **trained** pointer can outperform **frozen CE
argmin**; here the head is **untrained** and only checks **representation alignment** under restore.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tf_kv_incremental import IncrementalCausalTransformerKV
from .tf_kv_tree_nav import TfKvNavState, TfKvTruncateNavState, dfs_tf_kv_nav
from .tree import TreeNode, find_root_leaf_path_ending_at


def tf_kv_fixed_leaf_head_ce_nav_vs_gold_path(
    root: TreeNode,
    target_leaf: TreeNode,
    *,
    dim: int,
    tf_layers: int,
    nhead: int,
    ff_mult: int,
    dev: torch.device,
    use_truncate_restore: bool,
    num_leaves: int,
    target_leaf_index: int,
    probe_seed: int = 12_345,
) -> dict[str, object]:
    if not (0 <= target_leaf_index < num_leaves):
        raise ValueError("target_leaf_index out of range for num_leaves")

    model = IncrementalCausalTransformerKV(
        dim=dim, nhead=nhead, num_layers=tf_layers, ff_mult=ff_mult
    ).to(dev)
    model.eval()
    state: TfKvNavState | TfKvTruncateNavState
    if use_truncate_restore:
        state = TfKvTruncateNavState(model=model)
    else:
        state = TfKvNavState(model=model)

    with torch.no_grad():
        ok = dfs_tf_kv_nav(
            root,
            state,
            leaf_goal=lambda n: n is target_leaf,
            trace=None,
            mount_snapshot=None,
        )
    if not ok:
        return {"dfs_ok": False, "error": "dfs did not reach target_leaf"}

    h_nav = model.read_last_token_hidden()
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    gold = find_root_leaf_path_ending_at(root, target_leaf)

    ref = IncrementalCausalTransformerKV(
        dim=dim, nhead=nhead, num_layers=tf_layers, ff_mult=ff_mult
    ).to(dev)
    ref.load_state_dict({k: v.to(dev) for k, v in sd_cpu.items()})
    ref.eval()
    ref.reset()

    pos = 0
    with torch.no_grad():
        for node in gold:
            x = node.embedding.unsqueeze(0)
            ref.forward_chunk(x, pos_offset=pos)
            pos += int(x.shape[1])
    h_ref = ref.read_last_token_hidden()

    g = torch.Generator(device=dev)
    g.manual_seed(probe_seed)
    head = nn.Linear(dim, num_leaves, device=dev)
    nn.init.normal_(head.weight, std=0.02, generator=g)
    nn.init.zeros_(head.bias)

    y = torch.tensor([target_leaf_index], device=dev, dtype=torch.long)
    with torch.no_grad():
        logits_nav = head(h_nav.unsqueeze(0))
        logits_ref = head(h_ref.unsqueeze(0))
        ce_nav = float(F.cross_entropy(logits_nav, y).item())
        ce_ref = float(F.cross_entropy(logits_ref, y).item())
        max_logit_diff = float((logits_nav - logits_ref).abs().max().item())

    return {
        "dfs_ok": True,
        "arm": "truncate_kv_restore" if use_truncate_restore else "full_kv_clone_restore",
        "gold_path_nodes": len(gold),
        "probe_seed": probe_seed,
        "num_leaves": num_leaves,
        "target_leaf_index": target_leaf_index,
        "ce_nav": round(ce_nav, 8),
        "ce_ref": round(ce_ref, 8),
        "abs_ce_delta": round(abs(ce_nav - ce_ref), 10),
        "max_abs_logit_diff": round(max_logit_diff, 10),
    }
