#!/usr/bin/env bash
# 在 Linux / AutoDL（已装 torch + transformers + fused mamba_ssm）上跑与本地可对齐的扫参网格。
# 用法（仓库根目录）：
#   chmod +x scripts/benchmarks/run_server_sweep_aligned.sh
#   TAG=my3090 ./scripts/benchmarks/run_server_sweep_aligned.sh
#
# 输出默认在 results/metrics/；若设 MAMBA2_RESULTS_ROOT 则写入该目录下 metrics/。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

TAG="${TAG:-$(date +%Y%m%d_%H%M)}"
BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics"
mkdir -p "$OUT"

echo "=== repo: $ROOT  tag=$TAG  out=$OUT ==="
python scripts/smoke/smoke_local.py
python scripts/smoke/smoke_mamba_minimal.py --reps 2 --warmup 1

echo "=== A: preset local, dim=128, 8 点（与早期 local/云端对齐键）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset local \
  --dim 128 --warmup 2 --reps 8 --max-leaves 512 \
  --out-csv "$OUT/sweep_adl_dim128_localgrid_${TAG}.csv"

echo "=== B: dim=256, depth 3–6 × chunk 4,8,12, 12 点（对齐本地 extended）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 3,4,5,6 --chunk-lens 4,8,12 --fanout 2 --dim 256 --max-leaves 64 \
  --warmup 1 --reps 3 \
  --out-csv "$OUT/sweep_adl_dim256_chunk412_${TAG}.csv"

echo "=== C: dim=256, chunk 含 16, reps=8, 16 点 ==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 3,4,5,6 --chunk-lens 4,8,12,16 --fanout 2 --dim 256 --max-leaves 64 \
  --warmup 2 --reps 8 \
  --out-csv "$OUT/sweep_adl_dim256_chunk416_r8_${TAG}.csv"

echo "=== D: dim=384, depth 4–6 × chunk 8,16, 6 点 ==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 4,5,6 --chunk-lens 8,16 --fanout 2 --dim 384 --max-leaves 64 \
  --warmup 2 --reps 10 \
  --out-csv "$OUT/sweep_adl_dim384_${TAG}.csv"

echo "=== 完成。请下载: ==="
ls -la "$OUT/sweep_adl_"*"${TAG}.csv"
