#!/usr/bin/env python3
"""
Local smoke test (5060 8G friendly): PyTorch + optional CUDA + optional mamba_ssm.

Run from repo root:
  python scripts/smoke/smoke_local.py
"""
from __future__ import annotations

import os
import sys
import time


def _print_env_roots() -> None:
    for key in ("MAMBA2_DATA_ROOT", "MAMBA2_CKPT_ROOT", "MAMBA2_RESULTS_ROOT"):
        val = os.environ.get(key)
        print(f"{key}={val!r}" if val else f"{key}=(unset; see docs/environment/SYNC_AND_ENVIRONMENTS.md)")


def main() -> int:
    print("=== mamba2 local smoke ===")
    _print_env_roots()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed. Example: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return 1

    print(f"torch={torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda.is_available()={cuda}")
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        name = torch.cuda.get_device_name(0)
        print(f"device[0]={name}")
        print(f"torch.version.cuda={torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        print(f"total_memory_gib={props.total_memory / (1024**3):.2f}")
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"mem_free_gib={free_b / (1024**3):.2f} total_gib={total_b / (1024**3):.2f}")

    # Small forward: batch 4, seq 128, dim 256 — safe on 8G
    m = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 256),
    ).to(device=device, dtype=torch.float32)
    x = torch.randn(4, 128, 256, device=device, dtype=torch.float32)

    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 50
    for _ in range(reps):
        y = m(x)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"{reps}x forward (4,128,256) elapsed_s={elapsed:.4f} out_shape={tuple(y.shape)}")

    try:
        import mamba_ssm  # noqa: F401

        print(f"mamba_ssm: import OK (version getattr: {getattr(mamba_ssm, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"mamba_ssm: not installed (expected until you add SSM stack) — {e}")

    print("=== smoke OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
