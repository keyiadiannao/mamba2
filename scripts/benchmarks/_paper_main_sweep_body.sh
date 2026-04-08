#!/usr/bin/env bash
# 由 run_server_paper_main_sweep.sh / run_server_paper_main_sweep_naive.sh source。
# 前置：已 set -euo pipefail，已 cd 到仓库 ROOT，已设 WARMUP REPS TAG OUT；已写 manifest（naive 版在 source 前自检）。

python scripts/smoke/smoke_local.py
python scripts/smoke/smoke_mamba_minimal.py --warmup 1 --reps 2

echo "=== A: dim=256, depth 3–6 × chunk 4,8,12（12 点，主曲线）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 3,4,5,6 --chunk-lens 4,8,12 --fanout 2 --dim 256 --max-leaves 64 \
  --warmup "$WARMUP" --reps "$REPS" \
  --out-csv "$OUT/paper_main_dim256_${TAG}.csv"

echo "=== B: dim=128, preset local（8 点，与历史 localgrid 键一致）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset local \
  --dim 128 --max-leaves 512 --warmup "$WARMUP" --reps "$REPS" \
  --out-csv "$OUT/paper_main_dim128_localgrid_${TAG}.csv"

echo "=== C: dim=384, depth 4–6 × chunk 8,16（6 点）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 4,5,6 --chunk-lens 8,16 --fanout 2 --dim 384 --max-leaves 64 \
  --warmup "$WARMUP" --reps "$REPS" \
  --out-csv "$OUT/paper_main_dim384_${TAG}.csv"
