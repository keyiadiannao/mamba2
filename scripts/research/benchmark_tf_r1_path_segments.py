#!/usr/bin/env python3
"""
S2 / §7.2 **TF-R1（重算）**：与 S1 相同的玩具树、**单条根—叶路径**、**累积前缀** token 序列；
对每个边界 *k* 仅测 ``TransformerPathReader`` 的 **一次完整前向**（``nn.TransformerEncoder`` 全序列自注意力，
**不**使用 KV cache，等价于「回退后从根重算到该前缀」的 wall-clock 与峰值显存）。

与 ``benchmark_mamba2_cache_snapshot_segments.py`` 对齐默认：``depth=4``、``chunk_len=8``、``dim=128``、
``tf_layers=2``、``nhead=8``（与 ``run_tree_reader_benchmark`` 一致）。

  python scripts/research/benchmark_tf_r1_path_segments.py --device cuda
  python scripts/research/benchmark_tf_r1_path_segments.py --out-json results/metrics/tf_r1_path_segments_depth4.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.readers import TransformerPathReader
from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths


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


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak_mib(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mib(dev: torch.device) -> float:
    if dev.type != "cuda":
        return 0.0
    _sync(dev)
    return float(torch.cuda.max_memory_allocated()) / (1024**2)


def _segment_forward_stats(
    model: torch.nn.Module,
    x: torch.Tensor,
    dev: torch.device,
    warmup: int,
    reps: int,
) -> tuple[float, float]:
    """Mean forward wall-clock (ms) over `reps`, and peak allocated MiB during those forwards (CUDA only)."""
    model.eval()
    x = x.to(dev)
    model = model.to(dev)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    _sync(dev)
    _reset_peak_mib(dev)
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(x)
    _sync(dev)
    mean_ms = (time.perf_counter() - t0) / max(reps, 1) * 1000.0
    return mean_ms, _peak_mib(dev)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4, help="tree depth; path has depth+1 nodes")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if args.dim % args.nhead != 0:
        print("dim must be divisible by nhead", file=sys.stderr)
        return 1

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)

    root = build_balanced_tree(args.depth, args.fanout, args.chunk_len, args.dim, dev, torch.float32, gen)
    paths = list(iter_root_leaf_paths(root))
    path = paths[0]
    nodes = len(path)

    reader = TransformerPathReader(dim=args.dim, nhead=args.nhead, num_layers=args.tf_layers)
    reader.to(dev)

    per_seg: list[dict[str, float | int]] = []
    prefix_chunks: list[torch.Tensor] = []

    for seg_i, node in enumerate(path):
        prefix_chunks.append(node.embedding)
        cum = torch.cat(prefix_chunks, dim=0).unsqueeze(0)
        seq_len = int(cum.shape[1])
        mean_ms, peak_mib = _segment_forward_stats(reader, cum, dev, args.warmup, args.reps)
        per_seg.append(
            {
                "segment_index": seg_i,
                "nodes_on_path": nodes,
                "seq_len": seq_len,
                "forward_mean_ms": round(mean_ms, 6),
                "peak_alloc_mib": round(peak_mib, 4) if dev.type == "cuda" else 0.0,
            }
        )

    payload = {
        "kind": "tf_r1_path_segments",
        "baseline": "TF-R1",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "device": str(dev),
        "tree_depth_param": args.depth,
        "path_nodes": nodes,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "tf_layers": args.tf_layers,
        "nhead": args.nhead,
        "warmup": args.warmup,
        "reps": args.reps,
        "per_segment": per_seg,
        "note": (
            "Each row: one full TransformerPathReader forward on cumulative prefix [1, seq_len, dim]; "
            "no KV cache — §7.2 TF-R1 recompute cost at that tree boundary. Compare S1 Mamba cache clone JSON."
        ),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
