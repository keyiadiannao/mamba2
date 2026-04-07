#!/usr/bin/env python3
"""
Phase-1 stub: balanced tree, all root-to-leaf paths, compare Transformer vs GRU reader.

Mamba-2 can replace GRUPathReader later; this script establishes the harness (latency + VRAM).

Run (from repo root, conda env mamba2):
  python scripts/benchmark_tree_walk.py --depth 6 --fanout 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag_tree.benchmark_core import run_tree_reader_benchmark


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=6, help="Balanced k-ary tree depth (leaves = fanout**depth)")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8, help="Tokens per tree node (synthetic)")
    p.add_argument("--dim", type=int, default=128, help="Hidden size (128 divisible by nhead=8)")
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=str, default="", help="Write metrics JSON to this path")
    args = p.parse_args()

    import torch

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    results = run_tree_reader_benchmark(
        depth=args.depth,
        fanout=args.fanout,
        chunk_len=args.chunk_len,
        dim=args.dim,
        nhead=args.nhead,
        tf_layers=args.tf_layers,
        gru_layers=args.gru_layers,
        warmup=args.warmup,
        reps=args.reps,
        device=device,
        seed=args.seed,
    )

    print(json.dumps(results, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_root = os.environ.get("MAMBA2_RESULTS_ROOT", "")
        if results_root:
            results["results_root_env"] = results_root
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
