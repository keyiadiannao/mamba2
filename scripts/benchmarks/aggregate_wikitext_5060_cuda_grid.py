#!/usr/bin/env python3
"""
Flatten **5060 CUDA** Wikitext benchmark JSONs (2×2: n8/n16 × c8/c12) into one CSV.

  python scripts/benchmarks/aggregate_wikitext_5060_cuda_grid.py \\
    --glob 'results/metrics_result/benchmark_wikitext_5060_cuda_n*_c*_20260407.json' \\
    --out-csv results/metrics_result/benchmark_wikitext_5060_cuda_grid_20260407.csv

Requires filenames to contain ``n{num}_c{chunk}`` (e.g. ``n16_c12``).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KEY_RE = re.compile(r"n(\d+)_c(\d+)")


def _parse_nc(path: Path) -> tuple[int, int] | None:
    m = _KEY_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--glob",
        type=str,
        default="results/metrics_result/benchmark_wikitext_5060_cuda_n*_c*_20260407.json",
        help="glob under repo root",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/metrics_result/benchmark_wikitext_5060_cuda_grid_20260407.csv"),
        metavar="PATH",
    )
    args = p.parse_args()

    root = _REPO_ROOT
    paths = sorted(root.glob(args.glob))
    rows: list[dict[str, object]] = []
    for path in paths:
        nc = _parse_nc(path)
        if nc is None:
            print(f"skip (no n*_c* in name): {path}", file=sys.stderr)
            continue
        n_leaves, chunk_len = nc
        data = json.loads(path.read_text(encoding="utf-8"))
        git_sha = data.get("git_sha", "")
        device = data.get("device", "")
        for reader in ("transformer", "gru", "mamba2"):
            block = data.get(reader)
            if not isinstance(block, dict):
                continue
            rows.append(
                {
                    "json_file": str(path.relative_to(root)).replace("\\", "/"),
                    "num_leaves": n_leaves,
                    "chunk_len": chunk_len,
                    "reader": reader,
                    "peak_alloc_mib": block.get("peak_alloc_mib"),
                    "per_step_s": block.get("per_step_s"),
                    "elapsed_s": block.get("elapsed_s"),
                    "batch_paths": block.get("batch_paths"),
                    "tokens_per_path": block.get("tokens_per_path"),
                    "dim": block.get("dim"),
                    "device": device,
                    "git_sha": git_sha,
                }
            )

    rows.sort(key=lambda r: (r["num_leaves"], r["chunk_len"], str(r["reader"])))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("ERROR: no rows; check --glob", file=sys.stderr)
        return 1
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
