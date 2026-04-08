#!/usr/bin/env python3
"""
Probe HuggingFace ``Mamba2Model`` forward outputs (for §7.5 S1: what to snapshot).

Runs a tiny forward with ``inputs_embeds``; prints ``return_dict`` keys, shapes,
and cache-related fields. With ``--use-cache``, HF may attach ``cache_params`` (e.g.
``DynamicCache``). Without ``mamba_ssm``, expect a one-time stderr warning about
the fast path; fused env on AutoDL should show ``mamba_ssm True``.

**Note:** On **CUDA + fused**, this script uses **8 heads** (``head_dim=32`` for
``hidden=128``) and at least **batch 8** to avoid ``causal_conv1d`` stride errors
with tiny batches; ``Mamba2PathReader`` in benchmarks may use 4 heads but larger
path batch — both are valid probes of the same HF module family.

  python scripts/research/probe_mamba2_outputs.py
  python scripts/research/probe_mamba2_outputs.py --device cuda --use-cache
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=32, help="seq len; fused CUDA often needs multiple of 8")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--use-cache", action="store_true", help="call with use_cache=True (if supported)")
    args = p.parse_args()

    dev = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    has_ssm = importlib.util.find_spec("mamba_ssm") is not None
    has_cc = importlib.util.find_spec("causal_conv1d") is not None

    from transformers import Mamba2Config, Mamba2Model

    expand = 2
    inner = args.hidden * expand
    # Fused causal_conv1d on CUDA is picky about layout; prefer num_heads >= 8 (see state-spaces/mamba#643).
    head_dim = 32
    num_heads = inner // head_dim
    if inner % head_dim != 0:
        head_dim = 64
        num_heads = inner // head_dim
    cfg = Mamba2Config(
        num_hidden_layers=args.layers,
        hidden_size=args.hidden,
        state_size=16,
        vocab_size=32000,
        num_heads=num_heads,
        head_dim=head_dim,
        expand=expand,
        n_groups=1,
        use_cache=args.use_cache,
    )
    model = Mamba2Model(cfg).to(dev)
    model.eval()

    batch = args.batch
    if dev.type == "cuda" and has_ssm and batch < 8:
        print(
            "# note: batch bumped to 8 for fused causal_conv1d (batch=1 can hit stride error); "
            "use --batch 8 or larger explicitly.",
            file=sys.stderr,
            flush=True,
        )
        batch = 8

    x = torch.randn(batch, args.seq, args.hidden, device=dev, dtype=torch.float32).contiguous()

    print("device", dev, "mamba_ssm", has_ssm, "causal_conv1d", has_cc, flush=True)
    print("batch", batch, "seq", args.seq, "hidden", args.hidden, "num_heads", num_heads, "head_dim", head_dim, flush=True)
    with torch.no_grad():
        out = model(inputs_embeds=x, use_cache=args.use_cache, return_dict=True)

    print("output type:", type(out).__name__, flush=True)
    if hasattr(out, "keys"):
        for k in out.keys():
            v = getattr(out, k, None)
            if v is None:
                print(f"  {k}: None", flush=True)
            elif torch.is_tensor(v):
                print(f"  {k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}", flush=True)
            else:
                print(f"  {k}: {type(v).__name__} {repr(v)[:200]}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
