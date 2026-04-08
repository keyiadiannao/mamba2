#!/usr/bin/env python3
"""
SSGS 张量快照微基准（无 LM）：

1) 在真实树上跑 ``dfs_ssgs_tensor``，统计 trace。
2) 纯 ``clone`` + ``copy_`` 循环，估计单决策点快照/恢复的下限噪声。

  python scripts/benchmarks/benchmark_ssgs_tensor_overhead.py --dim 256 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.ssgs import SSGSTrace, TensorNavState, dfs_ssgs_tensor
from src.rag_tree.tree import TreeNode, build_balanced_tree, iter_root_leaf_paths


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--micro-iters", type=int, default=50_000, help="clone+restore loop count")
    args = p.parse_args()

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    root = build_balanced_tree(args.depth, args.fanout, args.chunk_len, args.dim, dev)
    leaves = [path[-1] for path in iter_root_leaf_paths(root)]
    target = leaves[-1]

    def leaf_goal(n: TreeNode) -> bool:
        return n is target

    state = TensorNavState.zeros(args.dim, dev)
    tr = SSGSTrace()
    _sync(dev)
    t0 = time.perf_counter()
    ok = dfs_ssgs_tensor(root, state, leaf_goal=leaf_goal, trace=tr)
    _sync(dev)
    nav_s = time.perf_counter() - t0

    h = torch.randn(args.dim, device=dev)
    snap = h.clone()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.micro_iters):
        x = h.clone()
        h.copy_(snap)
    _sync(dev)
    micro_s = time.perf_counter() - t0
    per_pair_ms = micro_s / args.micro_iters * 1000.0

    out = {
        "device": str(dev),
        "depth": args.depth,
        "fanout": args.fanout,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "nav_ok": ok,
        "nav_wall_s": round(nav_s, 6),
        "trace": {
            "snapshots_taken": tr.snapshots_taken,
            "rollbacks": tr.rollbacks,
            "leaf_checks": tr.leaf_checks,
        },
        "micro_clone_restore": {
            "iters": args.micro_iters,
            "total_s": round(micro_s, 6),
            "per_clone_plus_restore_ms": round(per_pair_ms, 6),
        },
    }
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
