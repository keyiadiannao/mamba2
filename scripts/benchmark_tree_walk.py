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
import time
from pathlib import Path

# allow `python scripts/foo.py` without installing package
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.readers import GRUPathReader, TransformerPathReader
from src.rag_tree.tree import batched_paths, build_balanced_tree


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_reader(name: str, model: torch.nn.Module, x: torch.Tensor, device: torch.device, warmup: int, reps: int) -> dict:
    model = model.to(device)
    x = x.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(warmup):
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad(set_to_none=True)

    _reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "reader": name,
        "elapsed_s": round(elapsed, 6),
        "per_step_s": round(elapsed / reps, 6),
        "peak_alloc_mib": round(_peak_mib(), 2),
        "batch_paths": int(x.shape[0]),
        "tokens_per_path": int(x.shape[1]),
        "dim": int(x.shape[2]),
    }


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
    p.add_argument("--out-json", type=str, default="", help="Write metrics JSON to this path")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    root = build_balanced_tree(
        args.depth, args.fanout, args.chunk_len, args.dim, device, torch.float32, gen
    )
    paths, num_paths = batched_paths(root)
    leaves = args.fanout**args.depth
    if num_paths != leaves:
        raise RuntimeError(f"path count {num_paths} != fanout**depth {leaves}")

    tf = TransformerPathReader(dim=args.dim, nhead=args.nhead, num_layers=args.tf_layers)
    gru = GRUPathReader(dim=args.dim, num_layers=args.gru_layers)

    results = {
        "device": str(device),
        "depth": args.depth,
        "fanout": args.fanout,
        "num_leaves": num_paths,
        "chunk_len": args.chunk_len,
        "transformer": benchmark_reader("transformer", tf, paths, device, args.warmup, args.reps),
        "gru": benchmark_reader("gru", gru, paths, device, args.warmup, args.reps),
    }

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
