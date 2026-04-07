#!/usr/bin/env python3
"""
Minimal local Mamba forward (and optional backward) using HuggingFace Transformers.

Without `mamba-ssm` / `causal-conv1d`, Transformers falls back to a **sequential**
PyTorch implementation (slower than fused kernels, but fine for 5060 8G smoke tests).

Requires: pip install "transformers>=4.45" accelerate

  python scripts/smoke_mamba_minimal.py
  python scripts/smoke_mamba_minimal.py --seq 256 --backward
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Reduce HF log noise; fused-kernel warning still prints once from modeling code.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import MambaConfig, MambaModel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=2, help="Mamba blocks (keep small on 8G)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--state-size", type=int, default=16, dest="state_size")
    p.add_argument("--intermediate", type=int, default=512, help="FFN inner dim")
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--backward", action="store_true", help="Run backward each step (needs more VRAM)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        print("CUDA not available; use --cpu or fix PyTorch GPU install.", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu else "cuda")
    cfg = MambaConfig(
        num_hidden_layers=args.layers,
        hidden_size=args.hidden,
        state_size=args.state_size,
        intermediate_size=args.intermediate,
        vocab_size=args.vocab,
        use_cache=False,
    )
    model = MambaModel(cfg).to(device)
    if args.backward:
        model.train()
    else:
        model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} config=L{args.layers} H{args.hidden} N{args.state_size}")

    input_ids = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.backward else None

    def step() -> None:
        out = model(input_ids)
        loss = out.last_hidden_state.float().mean()
        if args.backward and opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    with torch.no_grad() if not args.backward else torch.enable_grad():
        for _ in range(args.warmup):
            step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(args.reps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mib = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    )
    print(f"{args.reps}x forward{'+backward' if args.backward else ''} elapsed_s={elapsed:.4f} per_step_s={elapsed/args.reps:.4f}")
    if device.type == "cuda":
        print(f"peak_alloc_mib={peak_mib:.2f}")

    print("=== mamba minimal smoke OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
