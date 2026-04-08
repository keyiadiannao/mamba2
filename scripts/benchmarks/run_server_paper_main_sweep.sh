#!/usr/bin/env bash
# 同机「主文级」扫参：全程固定 WARMUP / REPS，便于论文主图与表格（避免跨机混用不同计时设定）。
# 预期环境：Linux / AutoDL，已激活 conda mamba2，torch CUDA OK；（可选）mamba_ssm fused 以得到 Mamba2 低峰值。
#
# 用法（仓库根目录）：
#   source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
#   chmod +x scripts/benchmarks/run_server_paper_main_sweep.sh
#   WARMUP=2 REPS=8 TAG=paper_main_v1 ./scripts/benchmarks/run_server_paper_main_sweep.sh
#
# 输出：results/metrics/paper_main_*_${TAG}.csv + paper_main_manifest_${TAG}.txt
# 若遇 bash\r：sed -i 's/\r$//' scripts/benchmarks/run_server_paper_main_sweep.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

WARMUP="${WARMUP:-2}"
REPS="${REPS:-8}"
TAG="${TAG:-paper_main_$(date +%Y%m%d_%H%M)}"
BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics"
mkdir -p "$OUT"

echo "=== paper main sweep  tag=$TAG  warmup=$WARMUP  reps=$REPS  out=$OUT ==="

MANIFEST="$OUT/paper_main_manifest_${TAG}.txt"
{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

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

echo "=== 完成 ==="
ls -la "$OUT/paper_main_"*"${TAG}"*
