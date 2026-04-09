#!/usr/bin/env python3
"""
**SSGS × Mamba** on the **same** Wikitext-2 shallow tree as ``benchmark_wikitext_tree.py``:
``wikitext2_leaf_chunks`` → ``build_bottom_up_text_tree`` → ``dfs_ssgs_mamba`` (token-step + ``DynamicCache``).

This bridges **path-batch reader** benchmarks and **state-snapshot DFS navigation** on **real corpus leaves**,
for the research thread: **state rollback + Mamba + tree RAG**.

CUDA: ``build_toy_mamba2_for_ssgs`` patches mixers to ``torch_forward`` (batch=1 token steps).

Examples::

  python scripts/research/demo_ssgs_mamba_wikitext.py --cpu --num-leaves 8 --target-leaf-index -1
  python scripts/research/demo_ssgs_mamba_wikitext.py --device cuda --num-leaves 8 --layers 2 \\
    --out-json results/metrics/ssgs_mamba_wikitext_n8_rightmost_cpu.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
from src.rag_tree.ssgs import (
    MambaNavState,
    SSGSTrace,
    build_toy_mamba2_for_ssgs,
    dfs_ssgs_mamba,
    mount_mamba_cache_meta_on_tree,
)
from src.rag_tree.tree import iter_root_leaf_paths


def _git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _depth_edges(num_leaves: int, fanout: int) -> int:
    d = math.log(num_leaves) / math.log(fanout)
    if abs(d - round(d)) > 1e-9:
        raise ValueError(f"num_leaves={num_leaves} must be fanout**depth for fanout={fanout}")
    return int(round(d))


def main() -> int:
    try:
        from transformers import Mamba2Model  # noqa: F401
    except ImportError:
        print("transformers with Mamba2Model required", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument(
        "--target-leaf-index",
        type=int,
        default=-1,
        help="left-to-right leaf order (0..n-1); -1 = rightmost (default)",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="write reproducibility JSON (events, counters, tree meta)",
    )
    args = p.parse_args()

    fanout = args.fanout
    n = args.num_leaves
    _depth_edges(n, fanout)

    try:
        leaves = wikitext2_leaf_chunks(n, args.chars_per_leaf, config=args.wikitext_config)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    want_cuda = args.device == "cuda" and not args.cpu
    dev = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")

    root = build_bottom_up_text_tree(leaves, fanout, args.chunk_len, args.dim, dev, torch.float32)
    leaves_ordered = [path[-1] for path in iter_root_leaf_paths(root)]
    if len(leaves_ordered) != n:
        raise RuntimeError(f"leaf count mismatch {len(leaves_ordered)} != {n}")
    tidx = args.target_leaf_index
    if tidx < 0:
        tidx += n
    if not (0 <= tidx < n):
        print(f"ERROR: target_leaf_index out of range after wrap: {tidx} (n={n})", file=sys.stderr)
        return 1
    target = leaves_ordered[tidx]

    def leaf_goal(node: object) -> bool:
        return node is target

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
    print(
        "ok",
        ok,
        "snapshots_taken",
        tr.snapshots_taken,
        "rollbacks",
        tr.rollbacks,
        "leaf_checks",
        tr.leaf_checks,
        "device",
        dev,
        "target_leaf_index",
        tidx,
    )

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        payload = {
            "kind": "ssgs_mamba_wikitext_tree",
            "git_sha": _git_short_sha(_REPO_ROOT),
            "device": str(dev),
            "tree_kind": "wikitext2_bottom_up",
            "dataset": "wikitext",
            "wikitext_config": args.wikitext_config,
            "num_leaves": n,
            "fanout": fanout,
            "chars_per_leaf": args.chars_per_leaf,
            "chunk_len": args.chunk_len,
            "dim": args.dim,
            "mamba_layers": args.layers,
            "target_leaf_index": tidx,
            "mamba_torch_forward_only": dev.type == "cuda",
            "ok": ok,
            "snapshots_taken": tr.snapshots_taken,
            "rollbacks": tr.rollbacks,
            "leaf_checks": tr.leaf_checks,
            "events": list(tr.events),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("wrote", out_path)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
