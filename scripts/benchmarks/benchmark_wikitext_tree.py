#!/usr/bin/env python3
"""
Shallow tree on **real** public text: Wikitext-2 (HF `datasets`) -> leaf chunks ->
bottom-up `TreeNode` -> same TF / GRU / Mamba2 path reader benchmark.

Requires: pip install datasets

  python scripts/benchmarks/benchmark_wikitext_tree.py --num-leaves 8 --fanout 2

AutoDL / 无法直连 huggingface.co 时，在运行前任选其一::

  export HF_ENDPOINT=https://hf-mirror.com
  # 或：export MAMBA2_USE_HF_MIRROR=1
  # 自定义镜像：export MAMBA2_HF_ENDPOINT=https://你的镜像主机

见 ``docs/environment/AUTODL_SETUP.md``（Hugging Face 镜像）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.benchmark_core import run_reader_benchmark_on_paths
from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
from src.rag_tree.tree import batched_paths


def _depth_edges(num_leaves: int, fanout: int) -> int:
    d = math.log(num_leaves) / math.log(fanout)
    if abs(d - round(d)) > 1e-9:
        raise ValueError(f"num_leaves={num_leaves} must be fanout**depth for fanout={fanout}")
    return int(round(d))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mamba2", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="reserved for future sampling")
    args = p.parse_args()
    _ = args.seed

    fanout = args.fanout
    n = args.num_leaves
    depth_edges = _depth_edges(n, fanout)

    try:
        leaves = wikitext2_leaf_chunks(n, args.chars_per_leaf, config=args.wikitext_config)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    root = build_bottom_up_text_tree(leaves, fanout, args.chunk_len, args.dim, device, torch.float32)
    paths, num_paths = batched_paths(root)
    if num_paths != n:
        raise RuntimeError(f"path mismatch {num_paths} != {n}")

    out = run_reader_benchmark_on_paths(
        paths,
        nhead=args.nhead,
        include_mamba2=not args.no_mamba2,
        warmup=args.warmup,
        reps=args.reps,
        device=device,
    )
    out["tree_kind"] = "wikitext2_bottom_up"
    out["dataset"] = "wikitext"
    out["wikitext_config"] = args.wikitext_config
    out["num_leaves"] = n
    out["fanout"] = fanout
    out["depth"] = depth_edges
    out["chars_per_leaf"] = args.chars_per_leaf
    out["chunk_len"] = args.chunk_len

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
