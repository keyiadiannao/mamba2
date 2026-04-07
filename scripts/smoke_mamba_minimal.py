#!/usr/bin/env python3
"""
Minimal local Mamba / Mamba-2 forward (optional backward) via HuggingFace Transformers.

Default is **Mamba2** (`Mamba2Model`). Use `--arch mamba` for original `MambaModel`.

Without `mamba-ssm` / `causal-conv1d`, Transformers falls back to a naive PyTorch path
(slower; fine for 5060 8G smoke).

Requires: pip install "transformers>=4.45" accelerate (Mamba2 needs a recent 5.x)

  python scripts/smoke_mamba_minimal.py
  python scripts/smoke_mamba_minimal.py --arch mamba
  python scripts/smoke_mamba_minimal.py --seq 256 --backward
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch


def _build_model_mamba2(
    *,
    layers: int,
    hidden: int,
    state_size: int,
    vocab: int,
    expand: int,
    num_heads: int,
    head_dim: int,
):
    from transformers import Mamba2Config, Mamba2Model

    if hidden * expand != num_heads * head_dim:
        raise ValueError(
            f"Mamba2 requires hidden_size * expand == num_heads * head_dim; "
            f"got {hidden}*{expand}={hidden * expand} vs {num_heads}*{head_dim}={num_heads * head_dim}"
        )
    cfg = Mamba2Config(
        num_hidden_layers=layers,
        hidden_size=hidden,
        state_size=state_size,
        vocab_size=vocab,
        num_heads=num_heads,
        head_dim=head_dim,
        expand=expand,
        use_cache=False,
    )
    return (
        Mamba2Model(cfg),
        f"Mamba2 L{layers} H{hidden} N{state_size} exp{expand} heads={num_heads}*{head_dim}",
    )


def _build_model_mamba1(
    *,
    layers: int,
    hidden: int,
    state_size: int,
    intermediate: int,
    vocab: int,
):
    from transformers import MambaConfig, MambaModel

    cfg = MambaConfig(
        num_hidden_layers=layers,
        hidden_size=hidden,
        state_size=state_size,
        intermediate_size=intermediate,
        vocab_size=vocab,
        use_cache=False,
    )
    return MambaModel(cfg), f"Mamba L{layers} H{hidden} N{state_size} I{intermediate}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arch",
        choices=("mamba2", "mamba"),
        default="mamba2",
        help="mamba2 = Mamba2Model (default); mamba = original MambaModel",
    )
    p.add_argument("--layers", type=int, default=2, help="SSM blocks (keep small on 8G)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--state-size", type=int, default=16, dest="state_size")
    p.add_argument("--intermediate", type=int, default=512, help="(mamba v1 only) FFN inner dim")
    p.add_argument("--expand", type=int, default=2, help="(mamba2 only) expansion factor")
    p.add_argument("--num-heads", type=int, default=8, dest="num_heads", help="(mamba2 only)")
    p.add_argument("--head-dim", type=int, default=64, dest="head_dim", help="(mamba2 only)")
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

    if args.arch == "mamba2":
        model, cfg_str = _build_model_mamba2(
            layers=args.layers,
            hidden=args.hidden,
            state_size=args.state_size,
            vocab=args.vocab,
            expand=args.expand,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )
    else:
        model, cfg_str = _build_model_mamba1(
            layers=args.layers,
            hidden=args.hidden,
            state_size=args.state_size,
            intermediate=args.intermediate,
            vocab=args.vocab,
        )

    model = model.to(device)
    if args.backward:
        model.train()
    else:
        model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"architecture={args.arch} device={device} params={n_params:,} {cfg_str}")

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
    print(
        f"{args.reps}x forward{'+backward' if args.backward else ''} "
        f"elapsed_s={elapsed:.4f} per_step_s={elapsed/args.reps:.4f}"
    )
    if device.type == "cuda":
        print(f"peak_alloc_mib={peak_mib:.2f}")

    print("=== mamba minimal smoke OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
