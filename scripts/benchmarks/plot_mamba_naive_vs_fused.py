#!/usr/bin/env python3
"""
在同一 (depth, fanout, num_leaves, chunk_len, dim) 网格上对比两份扫参 CSV 的 **Mamba2 峰值显存**（及可选 per_step）。

典型用法：本地 naive（5060 HF） vs 仓库内 fused（AutoDL + mamba_ssm）——**两机、reps 可能不同**，
图题须声明「控制变量=脚本与网格，非绝对同机对照」；论文主文应以 **同机复跑** 为准。

  python scripts/benchmarks/plot_mamba_naive_vs_fused.py \\
    --csv-a results/metrics/sweep_tree_reader_20260410_local5060.csv \\
    --label-a \"5060 HF naive (cu128)\" \\
    --csv-b results/metrics/sweep_autodl_fused.csv \\
    --label-b \"3090 fused mamba_ssm (cu126)\" \\
    --out results/metrics/figures/mamba_naive_vs_fused_peak.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _key(row: dict[str, str]) -> Tuple[int, int, int, int, int]:
    return (
        int(row["depth"]),
        int(row["fanout"]),
        int(row["num_leaves"]),
        int(row["chunk_len"]),
        int(row["dim"]),
    )


def _load(path: Path) -> Dict[Tuple[int, int, int, int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {_key(r): r for r in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-a", type=Path, required=True)
    p.add_argument("--csv-b", type=Path, required=True)
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--out", type=Path, default=Path("results/metrics/figures/mamba_naive_vs_fused_peak.png"))
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return 1

    da = _load(args.csv_a)
    db = _load(args.csv_b)
    keys = sorted(set(da) & set(db))
    if not keys:
        print("no overlapping grid keys between CSVs", flush=True)
        return 1

    xs = list(range(len(keys)))
    xtick = [f"L{k[2]}\nc{k[3]}" for k in keys]
    peak_a = [float(da[k]["m2_peak_mib"]) for k in keys]
    peak_b = [float(db[k]["m2_peak_mib"]) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.6), 4))
    ax.plot(xs, peak_a, "o-", label=args.label_a)
    ax.plot(xs, peak_b, "s-", label=args.label_b)
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick, fontsize=8)
    ax.set_ylabel("m2_peak_mib (CUDA max_memory_allocated)")
    ax.set_xlabel("grid point (num_leaves, chunk_len); fanout&depth implicit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Mamba2 peak memory: two runs (verify same commit/grid on one machine for paper)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} ({len(keys)} common grid points)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
