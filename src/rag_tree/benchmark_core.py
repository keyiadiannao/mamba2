"""Shared tree-walk reader benchmark (latency + peak VRAM). Used by CLI and sweep scripts."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from src.rag_tree.readers import GRUPathReader, TransformerPathReader
from src.rag_tree.tree import batched_paths, build_balanced_tree


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_allocated_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_reader(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
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
        "peak_alloc_mib": round(peak_allocated_mib(), 2),
        "batch_paths": int(x.shape[0]),
        "tokens_per_path": int(x.shape[1]),
        "dim": int(x.shape[2]),
    }


def run_tree_reader_benchmark(
    *,
    depth: int,
    fanout: int,
    chunk_len: int,
    dim: int,
    nhead: int = 8,
    tf_layers: int = 2,
    gru_layers: int = 2,
    warmup: int = 3,
    reps: int = 10,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if dim % nhead != 0:
        raise ValueError("dim must be divisible by nhead")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    root = build_balanced_tree(depth, fanout, chunk_len, dim, dev, torch.float32, gen)
    paths, num_paths = batched_paths(root)
    leaves = fanout**depth
    if num_paths != leaves:
        raise RuntimeError(f"path count {num_paths} != fanout**depth {leaves}")

    tf = TransformerPathReader(dim=dim, nhead=nhead, num_layers=tf_layers)
    gru = GRUPathReader(dim=dim, num_layers=gru_layers)

    return {
        "device": str(dev),
        "depth": depth,
        "fanout": fanout,
        "num_leaves": num_paths,
        "chunk_len": chunk_len,
        "transformer": benchmark_reader("transformer", tf, paths, dev, warmup, reps),
        "gru": benchmark_reader("gru", gru, paths, dev, warmup, reps),
    }


def run_reader_benchmark_on_paths(
    paths: torch.Tensor,
    *,
    nhead: int = 8,
    tf_layers: int = 2,
    gru_layers: int = 2,
    warmup: int = 3,
    reps: int = 10,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Run the same reader timing on an explicit path batch [B, T, D] (e.g. text-shaped tree).
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = paths.to(dev)
    dim = int(paths.shape[2])
    if dim % nhead != 0:
        raise ValueError("dim must be divisible by nhead")

    tf = TransformerPathReader(dim=dim, nhead=nhead, num_layers=tf_layers)
    gru = GRUPathReader(dim=dim, num_layers=gru_layers)

    return {
        "device": str(dev),
        "depth": None,
        "fanout": None,
        "num_leaves": int(paths.shape[0]),
        "chunk_len": None,
        "transformer": benchmark_reader("transformer", tf, paths, dev, warmup, reps),
        "gru": benchmark_reader("gru", gru, paths, dev, warmup, reps),
    }
