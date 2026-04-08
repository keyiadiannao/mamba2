#!/usr/bin/env python3
"""
Plot `sweep_tree_benchmark.py` CSV: per-step latency and Mamba2 peak vs num_leaves,
one facet per chunk_len (avoids duplicate x when the same leaf count uses different chunk_len).

  python scripts/benchmarks/plot_tree_reader_sweep.py \\
    --csv results/metrics/sweep_tree_reader_20260410_local5060.csv \\
    --out results/metrics/figures/sweep_readers_20260410.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _f(row: dict[str, str], key: str) -> float:
    v = (row.get(key) or "").strip()
    if not v:
        return float("nan")
    return float(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/metrics/figures/sweep_readers.png"))
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", flush=True)
        return 1

    rows: list[dict[str, str]] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty csv", flush=True)
        return 1

    chunk_lens = sorted({int(r["chunk_len"]) for r in rows})
    fig, axes = plt.subplots(2, len(chunk_lens), figsize=(4 * len(chunk_lens), 6), squeeze=False)

    for j, cl in enumerate(chunk_lens):
        sub = [r for r in rows if int(r["chunk_len"]) == cl]
        sub.sort(key=lambda r: int(r["num_leaves"]))
        xs = [int(r["num_leaves"]) for r in sub]
        ax0 = axes[0, j]
        ax1 = axes[1, j]
        ax0.plot(xs, [_f(r, "tf_per_step_s") for r in sub], "o-", label="Transformer")
        ax0.plot(xs, [_f(r, "gru_per_step_s") for r in sub], "s-", label="GRU")
        ax0.plot(xs, [_f(r, "m2_per_step_s") for r in sub], "^-", label="Mamba2")
        ax0.set_xlabel("num_leaves (batch paths)")
        ax0.set_ylabel("per_step_s")
        ax0.set_title(f"chunk_len={cl}")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8)

        ax1.plot(xs, [_f(r, "tf_peak_mib") for r in sub], "o-", label="TF peak MiB")
        ax1.plot(xs, [_f(r, "m2_peak_mib") for r in sub], "^-", color="C2", label="Mamba2 peak MiB")
        ax1.set_xlabel("num_leaves (batch paths)")
        ax1.set_ylabel("peak_alloc_mib")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

    fig.suptitle(f"Tree reader sweep — {args.csv.name}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
