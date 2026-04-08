#!/usr/bin/env python3
"""
Sweep tree + reader benchmark grid; write CSV (and optional JSONL).

  python scripts/benchmarks/sweep_tree_benchmark.py --preset local --out-csv results/metrics/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.benchmark_core import run_tree_reader_benchmark


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


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _torch_and_gpu(device: torch.device) -> tuple[str, str]:
    ver = torch.__version__
    if device.type == "cuda" and torch.cuda.is_available():
        return ver, torch.cuda.get_device_name(0)
    return ver, ""


def iter_grid(
    depths: Iterable[int],
    chunk_lens: Iterable[int],
    fanout: int,
    max_leaves: int,
) -> list[tuple[int, int, int]]:
    combos: list[tuple[int, int, int]] = []
    for d, c in product(depths, chunk_lens):
        leaves = fanout**d
        if leaves > max_leaves:
            continue
        combos.append((d, c, leaves))
    return combos


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("none", "local"), default="none", help="local: small grid for 5060")
    p.add_argument("--depths", type=str, default="3,4,5,6", help="Comma-separated depths")
    p.add_argument("--chunk-lens", type=str, default="4,8", help="Comma-separated chunk lengths")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--no-mamba2", action="store_true")
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--mamba-hidden", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-leaves", type=int, default=512, help="Skip combos with fanout**depth above this")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--out-jsonl", type=str, default="", help="Append one JSON object per line")
    args = p.parse_args()

    if args.preset == "local":
        depths = [3, 4, 5, 6]
        chunk_lens = [4, 8]
        fanout = 2
        max_leaves = min(args.max_leaves, 128)
    else:
        depths = _parse_int_list(args.depths)
        chunk_lens = _parse_int_list(args.chunk_lens)
        fanout = args.fanout
        max_leaves = args.max_leaves

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch_version, gpu_name = _torch_and_gpu(device)
    repo = _REPO_ROOT
    sha = _git_short_sha(repo)

    grid = iter_grid(depths, chunk_lens, fanout, max_leaves)
    if not grid:
        print("No (depth, chunk_len) combos after max-leaves filter.", file=sys.stderr)
        return 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    jsonl_f = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None

    fieldnames = [
        "utc_iso",
        "git_sha",
        "device",
        "gpu_name",
        "torch_version",
        "depth",
        "fanout",
        "num_leaves",
        "chunk_len",
        "tokens_per_path",
        "dim",
        "warmup",
        "reps",
        "tf_elapsed_s",
        "tf_per_step_s",
        "tf_peak_mib",
        "gru_elapsed_s",
        "gru_per_step_s",
        "gru_peak_mib",
        "m2_elapsed_s",
        "m2_per_step_s",
        "m2_peak_mib",
    ]

    try:
        with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            w.writeheader()

            for depth, chunk_len, num_leaves in grid:
                utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                r = run_tree_reader_benchmark(
                    depth=depth,
                    fanout=fanout,
                    chunk_len=chunk_len,
                    dim=args.dim,
                    nhead=args.nhead,
                    tf_layers=args.tf_layers,
                    gru_layers=args.gru_layers,
                    include_mamba2=not args.no_mamba2,
                    mamba_layers=args.mamba_layers,
                    mamba_hidden=args.mamba_hidden,
                    warmup=args.warmup,
                    reps=args.reps,
                    device=device,
                    seed=args.seed,
                )
                tf = r["transformer"]
                gru = r["gru"]
                m2 = r.get("mamba2") or {}
                row = {
                    "utc_iso": utc,
                    "git_sha": sha,
                    "device": r["device"],
                    "gpu_name": gpu_name,
                    "torch_version": torch_version,
                    "depth": depth,
                    "fanout": fanout,
                    "num_leaves": num_leaves,
                    "chunk_len": chunk_len,
                    "tokens_per_path": tf["tokens_per_path"],
                    "dim": args.dim,
                    "warmup": args.warmup,
                    "reps": args.reps,
                    "tf_elapsed_s": tf["elapsed_s"],
                    "tf_per_step_s": tf["per_step_s"],
                    "tf_peak_mib": tf["peak_alloc_mib"],
                    "gru_elapsed_s": gru["elapsed_s"],
                    "gru_per_step_s": gru["per_step_s"],
                    "gru_peak_mib": gru["peak_alloc_mib"],
                    "m2_elapsed_s": m2.get("elapsed_s", ""),
                    "m2_per_step_s": m2.get("per_step_s", ""),
                    "m2_peak_mib": m2.get("peak_alloc_mib", ""),
                }
                w.writerow(row)
                fcsv.flush()

                if jsonl_f:
                    payload = {**row, "full": r}
                    jsonl_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    jsonl_f.flush()
    finally:
        if jsonl_f:
            jsonl_f.close()

    print(f"wrote {len(grid)} rows -> {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
