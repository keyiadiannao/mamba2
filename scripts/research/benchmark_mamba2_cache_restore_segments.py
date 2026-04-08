#!/usr/bin/env python3
"""
S4 / §7.3 **SSM restore**：与 S1 相同的玩具树、单路径、累积前缀前向得到 ``DynamicCache`` 后，将**已保存快照**
写回**当前** ``conv_states`` / ``recurrent_states``（``tensor.copy_(snapshot)``），统计 **restore_wall_ms**
与字节数（与 S1 ``clone_nbytes`` 同阶）。

- ``--snapshot-device same``（默认）：快照与模型同设备；restore ≈ 设备内 ``copy_``（带宽下界对照 S1 clone）。
- ``--snapshot-device cpu``：快照留在 CPU；每次 restore 为 ``live.copy_(snap.to(device))``，对应 §7.3「拷贝快照到设备」
  + 写回显存（**不含**重新跑 Mamba 前向）。

  python scripts/research/benchmark_mamba2_cache_restore_segments.py --device cuda
  python scripts/research/benchmark_mamba2_cache_restore_segments.py --device cuda --snapshot-device cpu
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


def _zero_live_cache(cache: object) -> None:
    layers = getattr(cache, "layers", None) or []
    for layer in layers:
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                t.zero_()


def _restore_from_snapshot(
    cache: object,
    snapshot: list[dict[str, torch.Tensor]],
    dev: torch.device,
    snapshot_on_cpu: bool,
) -> None:
    layers = getattr(cache, "layers", None) or []
    for li, layer in enumerate(layers):
        for name in ("conv_states", "recurrent_states"):
            live = getattr(layer, name, None)
            if not torch.is_tensor(live):
                continue
            snap = snapshot[li][name]
            if snapshot_on_cpu:
                live.copy_(snap.to(dev, dtype=live.dtype, non_blocking=False))
            else:
                live.copy_(snap)


def _mean_restore_ms(
    cache: object,
    snapshot: list[dict[str, torch.Tensor]],
    dev: torch.device,
    snapshot_on_cpu: bool,
    warmup: int,
    reps: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        _zero_live_cache(cache)
        _restore_from_snapshot(cache, snapshot, dev, snapshot_on_cpu)
    _sync(dev)
    _reset_peak_mib(dev)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        _zero_live_cache(cache)
        _restore_from_snapshot(cache, snapshot, dev, snapshot_on_cpu)
    _sync(dev)
    mean_ms = (time.perf_counter() - t0) / max(reps, 1) * 1000.0
    return mean_ms, _peak_mib(dev)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--snapshot-device",
        type=str,
        choices=("same", "cpu"),
        default="same",
        help="where snapshot tensors live before each restore",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    snapshot_on_cpu = args.snapshot_device == "cpu"
    if snapshot_on_cpu and dev.type != "cuda":
        print("--snapshot-device cpu is only meaningful with --device cuda", file=sys.stderr)
        return 1

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

    per_seg: list[dict[str, float | int | str]] = []
    prefix_chunks: list[torch.Tensor] = []

    for seg_i, node in enumerate(path):
        prefix_chunks.append(node.embedding)
        cum = torch.cat(prefix_chunks, dim=0).unsqueeze(0)
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

        nbytes = _cache_tensor_nbytes(cache_params)
        snap = _clone_cache_tensors(cache_params)
        if snapshot_on_cpu:
            snap = [
                {k: v.detach().cpu().contiguous() for k, v in block.items()} for block in snap
            ]

        restore_ms, peak_mib = _mean_restore_ms(
            cache_params, snap, dev, snapshot_on_cpu, args.warmup, args.reps
        )

        per_seg.append(
            {
                "segment_index": seg_i,
                "nodes_on_path": nodes,
                "seq_len": (seg_i + 1) * args.chunk_len,
                "restore_nbytes": nbytes,
                "restore_wall_ms": round(restore_ms, 6),
                "peak_alloc_mib": round(peak_mib, 4) if dev.type == "cuda" else 0.0,
            }
        )

    payload = {
        "kind": "mamba2_cache_restore_segments",
        "baseline": "SSM-restore",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "device": str(dev),
        "snapshot_device": args.snapshot_device,
        "tree_depth_param": args.depth,
        "path_nodes": nodes,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "mamba_layers": args.layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "warmup": args.warmup,
        "reps": args.reps,
        "per_segment": per_seg,
        "note": (
            "After S1-style cumulative forward, snapshot = clone(conv_states, recurrent_states); "
            "each trial: zero live cache then copy_ snapshot back. §7.3 restore_wall_ms (no re-encode). "
            "snapshot-device cpu includes H2D in copy_ path."
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
