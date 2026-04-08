#!/usr/bin/env python3
"""
S1：沿**单条根—叶路径**，在读完第 *k* 个节点后（累积序列长度 ``(k+1)*chunk_len``）对 ``Mamba2Model``
做一次**完整前向**（``use_cache=True``，``cache_params=None``），在每步后对 ``DynamicCache`` 内
``conv_states`` / ``recurrent_states`` 做 **clone**，统计 **nbytes** 与 **wall-clock**
（§7.5 S1 / ``RESEARCH_NOTES`` §7.1）。

**为何不用段间传 cache**：HF Mamba2 在 ``cache_params.has_previous_state`` 为真时走 fused 单步解码，
要求 ``seq_len == 1``；若下一段仍喂 ``[1, chunk_len, dim]`` 会触发 ``causal_conv1d_update`` 的
``weight must have shape (dim, width)``。累积整段前向与逐步因果消费在最终 cache 上等价，且 CUDA/CPU 一致。

与 ``Mamba2PathReader`` 一致：``inputs_embeds`` 宽度 = path ``dim``；内部 ``hidden_size=dim``，
``num_heads=8``、``head_dim=32``（fused CUDA 下与 ``probe_mamba2_outputs`` 对齐）。

  python scripts/research/benchmark_mamba2_cache_snapshot_segments.py --device cuda
  python scripts/research/benchmark_mamba2_cache_snapshot_segments.py --out-json results/metrics/mamba2_cache_snap_segments.json
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

from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths, path_tensor


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


def _cache_tensor_nbytes(cache: object) -> int:
    total = 0
    layers = getattr(cache, "layers", None) or []
    for layer in layers:
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                total += int(t.numel() * t.element_size())
    return total


def _clone_cache_tensors(cache: object) -> list[dict[str, torch.Tensor]]:
    """Deep clone all cache tensors (CPU or GPU copies)."""
    out: list[dict[str, torch.Tensor]] = []
    layers = getattr(cache, "layers", None) or []
    for layer in layers:
        block: dict[str, torch.Tensor] = {}
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                block[name] = t.clone().detach()
        out.append(block)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4, help="tree depth (edges root→leaf); path has depth+1 nodes")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)

    root = build_balanced_tree(args.depth, args.fanout, args.chunk_len, args.dim, dev, torch.float32, gen)
    paths = list(iter_root_leaf_paths(root))
    path = paths[0]
    nodes = len(path)

    expand = 2
    inner = args.dim * expand
    head_dim = 32
    num_heads = inner // head_dim
    if inner % head_dim != 0:
        head_dim = 64
        num_heads = inner // head_dim

    from transformers import Mamba2Config, Mamba2Model

    cfg = Mamba2Config(
        num_hidden_layers=args.layers,
        hidden_size=args.dim,
        state_size=16,
        vocab_size=32000,
        num_heads=num_heads,
        head_dim=head_dim,
        expand=expand,
        n_groups=1,
        use_cache=True,
    )
    model = Mamba2Model(cfg).to(dev)
    model.eval()

    per_seg: list[dict[str, float | int]] = []
    prefix_chunks: list[torch.Tensor] = []

    for seg_i, node in enumerate(path):
        prefix_chunks.append(node.embedding)
        cum = torch.cat(prefix_chunks, dim=0).unsqueeze(0)  # [1, (seg_i+1)*chunk_len, dim]
        _sync(dev)
        with torch.no_grad():
            out = model(
                inputs_embeds=cum,
                cache_params=None,
                use_cache=True,
                return_dict=True,
            )
        cache_params = out.cache_params
        if cache_params is None:
            print("no cache_params after segment", seg_i, file=sys.stderr)
            return 1

        view_bytes = _cache_tensor_nbytes(cache_params)
        _sync(dev)
        t0 = time.perf_counter()
        _clone_cache_tensors(cache_params)
        _sync(dev)
        clone_s = time.perf_counter() - t0
        clone_bytes = view_bytes  # same element count as view

        per_seg.append(
            {
                "segment_index": seg_i,
                "nodes_on_path": nodes,
                "view_cache_nbytes": view_bytes,
                "clone_wall_ms": round(clone_s * 1000.0, 6),
                "clone_nbytes": clone_bytes,
            }
        )

    payload = {
        "kind": "mamba2_cache_snapshot_segments",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "device": str(dev),
        "tree_depth_param": args.depth,
        "path_nodes": nodes,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "mamba_layers": args.layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "per_segment": per_seg,
        "total_clone_nbytes_sum_segments": sum(s["clone_nbytes"] for s in per_seg),
        "note": (
            "Each row: full cache after consuming root→node_k (one full forward on prefix; "
            "no cross-segment cache_params — required for HF fused Mamba2 when chunk_len>1). "
            "Sum of clone_nbytes is per-boundary reporting, not a single snapshot."
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
