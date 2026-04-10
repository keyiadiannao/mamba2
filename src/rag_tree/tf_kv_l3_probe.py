"""
Minimal **L3** hook for TF-KV tree navigation: DFS-with-restore vs gold-path-only forward.

Compares **last-token trunk hidden** after a successful ``dfs_tf_kv_nav`` to a **reference**
run that **only** applies ``forward_chunk`` along the root—target leaf path with the **same
weights**. **Cosine ≈ 1** indicates restore/truncate semantics preserve the prefix state.
This is **not** a downstream CE/LM probe; see M1 doc for full L3 semantics scope.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .tf_kv_incremental import IncrementalCausalTransformerKV
from .tf_kv_tree_nav import TfKvNavState, TfKvTruncateNavState, dfs_tf_kv_nav
from .tree import TreeNode, find_root_leaf_path_ending_at


def tf_kv_hidden_consistency_nav_vs_gold_path(
    root: TreeNode,
    target_leaf: TreeNode,
    *,
    dim: int,
    tf_layers: int,
    nhead: int,
    ff_mult: int,
    dev: torch.device,
    use_truncate_restore: bool,
) -> dict[str, object]:
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

    h_nav_f = h_nav.float()
    h_ref_f = h_ref.float()
    cos_t = F.cosine_similarity(h_nav_f.unsqueeze(0), h_ref_f.unsqueeze(0), dim=-1)
    cos = float(cos_t.squeeze().item())
    l2 = float((h_nav_f - h_ref_f).norm().item())
    return {
        "dfs_ok": True,
        "arm": "truncate_kv_restore" if use_truncate_restore else "full_kv_clone_restore",
        "gold_path_nodes": len(gold),
        "cosine_last_token_hidden": round(cos, 8),
        "l2_diff_last_token_hidden": round(l2, 8),
    }
