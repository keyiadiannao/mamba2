#!/usr/bin/env python3
"""
Text-shaped balanced tree: leaf lines from file (or built-in sample) -> bottom-up TreeNode -> same reader benchmark.

Embeddings are deterministic from text (SHA256-seeded RNG), not a neural encoder.

  python scripts/benchmark_text_tree.py --leaf-file experiments/A-20260408-text-shaped-tree/leaves_sample.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.benchmark_core import run_reader_benchmark_on_paths
from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.tree import batched_paths

DEFAULT_LEAF_LINES = [
    "RAG combines retrieval with generation.",
    "Tree RAG uses hierarchical summaries.",
    "Mamba uses a recurrent state for long sequences.",
    "Transformers use quadratic attention in length.",
    "状态空间模型压缩历史为固定维度。",
    "检索头预测是否需要查外部知识。",
    "实验在5060与AutoDL上分工运行。",
    "复现依赖Git与环境锁定文件。",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--leaf-file", type=str, default="", help="UTF-8 text, one leaf passage per line")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mamba2", action="store_true")
    args = p.parse_args()

    if args.leaf_file:
        raw = Path(args.leaf_file).read_text(encoding="utf-8")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        lines = list(DEFAULT_LEAF_LINES)

    n = len(lines)
    f = args.fanout
    cur = 1
    depth_edges = 0
    while cur < n:
        cur *= f
        depth_edges += 1
    if cur != n:
        print(
            f"ERROR: leaf count {n} must equal fanout**depth (fanout={f}); got up to {cur}.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    root = build_bottom_up_text_tree(lines, f, args.chunk_len, args.dim, device, torch.float32)
    paths, num_paths = batched_paths(root)
    if num_paths != n:
        raise RuntimeError(f"unexpected path count {num_paths} != {n}")

    out = run_reader_benchmark_on_paths(
        paths,
        nhead=args.nhead,
        include_mamba2=not args.no_mamba2,
        warmup=args.warmup,
        reps=args.reps,
        device=device,
    )
    out["tree_kind"] = "text_shaped_bottom_up"
    out["fanout"] = f
    out["depth"] = depth_edges
    out["chunk_len"] = args.chunk_len
    out["leaf_file"] = str(args.leaf_file) if args.leaf_file else "(defaults)"

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
