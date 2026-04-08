#!/usr/bin/env python3
"""
演示 **SSGS + HF Mamba2Model**：``dfs_ssgs_mamba`` 在玩具平衡树上 DFS，目标为**最右叶**；
内部节点打 ``DynamicCache`` 快照，兄弟回溯前 ``restore``（与 ``RESEARCH_NOTES`` §7 / ``ssgs.py`` 一致）。

使用 **按 token** 前向。CUDA 上自动 **patch 为 HF ``torch_forward``**（避免 fused ``causal_conv1d`` 在 batch=1 的 stride 限制）。需 ``transformers``（Mamba2）。

  python scripts/research/demo_ssgs_mamba_dfs.py --device cpu
  python scripts/research/demo_ssgs_mamba_dfs.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.ssgs import (
    MambaNavState,
    SSGSTrace,
    build_toy_mamba2_for_ssgs,
    dfs_ssgs_mamba,
    mount_mamba_cache_meta_on_tree,
)
from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths


def main() -> int:
    try:
        from transformers import Mamba2Model  # noqa: F401
    except ImportError:
        print("transformers with Mamba2Model required", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=2)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)

    root = build_balanced_tree(args.depth, args.fanout, args.chunk_len, args.dim, dev, torch.float32, gen)
    leaves = [path[-1] for path in iter_root_leaf_paths(root)]
    target = leaves[-1]

    def leaf_goal(n) -> bool:
        return n is target

    model = build_toy_mamba2_for_ssgs(args.dim, dev, num_layers=args.layers)
    state = MambaNavState(model=model)
    tr = SSGSTrace()
    ok = dfs_ssgs_mamba(
        root,
        state,
        leaf_goal=leaf_goal,
        trace=tr,
        mount_snapshot=mount_mamba_cache_meta_on_tree,
    )
    print("ok", ok, "snapshots_taken", tr.snapshots_taken, "rollbacks", tr.rollbacks, "leaf_checks", tr.leaf_checks)
    print("events", tr.events[:20], "..." if len(tr.events) > 20 else "")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
