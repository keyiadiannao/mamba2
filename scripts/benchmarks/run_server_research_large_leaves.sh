#!/usr/bin/env bash
# 云端研究扫参（fused）：**大叶数**（depth 6–7，最多 128 条路径），与主文 paper_main（max-leaves 64）区分。
# 需：Linux、conda activate mamba2、已装 mamba_ssm（主环境）；可选 HF 镜像环境变量。
#
#   export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
#   TAG=research_lg_v1 WARMUP=2 REPS=5 ./scripts/benchmarks/run_server_research_large_leaves.sh
#
# 产出：metrics/sweep_research_large_leaves_dim{128,256}_${TAG}.csv + manifest

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

WARMUP="${WARMUP:-2}"
REPS="${REPS:-5}"
TAG="${TAG:-research_large_leaves_$(date +%Y%m%d_%H%M)}"
BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics"
mkdir -p "$OUT"

MANIFEST="$OUT/sweep_research_large_leaves_manifest_${TAG}.txt"
{
  echo "kind=research_large_leaves_fused"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

python scripts/smoke/smoke_local.py

echo "=== A: dim=128, depth 6–7 × chunk 4,8（≤128 叶，4 格点）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 6,7 --chunk-lens 4,8 --fanout 2 --dim 128 --max-leaves 256 \
  --warmup "$WARMUP" --reps "$REPS" \
  --out-csv "$OUT/sweep_research_large_leaves_dim128_${TAG}.csv"

echo "=== B: dim=256, depth 6–7 × chunk 4,8（4 格点，更重）==="
python scripts/benchmarks/sweep_tree_benchmark.py --preset none \
  --depths 6,7 --chunk-lens 4,8 --fanout 2 --dim 256 --max-leaves 256 \
  --warmup "$WARMUP" --reps "$REPS" \
  --out-csv "$OUT/sweep_research_large_leaves_dim256_${TAG}.csv"

echo "=== 完成 ==="
ls -la "$OUT/sweep_research_large_leaves_"*"${TAG}"*
