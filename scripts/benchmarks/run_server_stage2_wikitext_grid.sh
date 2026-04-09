#!/usr/bin/env bash
# 阶段 2 **A2-S2**：Wikitext-2 浅树、**fused** `mamba2` 环境，**path-batch** 与 **5060 naive 四格**同拓扑
# （num_leaves × chunk_len），便于正文 **分列**对照（**禁止无脚注与 5060 混表**）。
#
# 前置：Linux、`conda activate mamba2`、已装 **mamba_ssm**；Hub 不可达时见 **AUTODL_SETUP.md** §2b。
#
# 默认跑满 **四格**（与 **PHASE2_DRAFT** §1.1 / 5060 动机一致）。省时用 **GRID=minimal**（3 格，跳过 n=8,c=12）。
#
# 一键（推荐）::
#
#   cd /path/to/mamba2 && git pull
#   find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'   # 若遇 bash\r
#   chmod +x scripts/benchmarks/run_server_stage2_wikitext_grid.sh
#   source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
#   export HF_ENDPOINT=https://hf-mirror.com   # 按需
#   export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results    # 可选
#   ./scripts/benchmarks/run_server_stage2_wikitext_grid.sh
#
# 环境变量::
#   WARMUP   默认 2（与 **paper_main** 一致）
#   REPS     默认 8（与 **paper_main** 同量级）；若要与 **5060 四格**完全同 repetitions，设 **REPS=5**
#   TAG      输出文件名前缀，默认 **stage2_fused**
#   STAMP    UTC 时间戳；默认 **date -u +%Y%m%dT%H%MZ**
#   GRID     **full**（默认，4 格）或 **minimal**（3 格：8×8、16×8、16×12）
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

WARMUP="${WARMUP:-2}"
REPS="${REPS:-8}"
TAG="${TAG:-stage2_fused}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
GRID="${GRID:-full}"

BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics_result"
mkdir -p "$OUT"

MANIFEST="$OUT/benchmark_wikitext_${TAG}_manifest_${STAMP}.txt"
{
  echo "kind=stage2_wikitext_fused_grid"
  echo "grid=$GRID"
  echo "tag=$TAG"
  echo "stamp=$STAMP"
  echo "warmup=$WARMUP reps=$REPS"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "HF_ENDPOINT=${HF_ENDPOINT:-}"
  echo "MAMBA2_USE_HF_MIRROR=${MAMBA2_USE_HF_MIRROR:-}"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

run_cell() {
  local n=$1 c=$2
  local json_path="$OUT/benchmark_wikitext_${TAG}_${STAMP}_n${n}_c${c}.json"
  echo "=== num_leaves=$n chunk_len=$c -> $json_path ==="
  python scripts/benchmarks/benchmark_wikitext_tree.py \
    --num-leaves "$n" --chunk-len "$c" --fanout 2 --dim 128 \
    --warmup "$WARMUP" --reps "$REPS" \
    --out-json "$json_path"
}

if [[ "$GRID" == "minimal" ]]; then
  run_cell 8 8
  run_cell 16 8
  run_cell 16 12
else
  run_cell 8 8
  run_cell 8 12
  run_cell 16 8
  run_cell 16 12
fi

CSV_OUT="$OUT/benchmark_wikitext_${TAG}_grid_${STAMP}.csv"
echo "=== aggregate -> $CSV_OUT ==="
python scripts/benchmarks/aggregate_wikitext_tree_json_grid.py \
  --base-dir "$BASE_OUT" \
  --glob "metrics_result/benchmark_wikitext_${TAG}_${STAMP}_n*_c*.json" \
  --out-csv "$CSV_OUT"

echo "=== done ==="
ls -la "$OUT/benchmark_wikitext_${TAG}"*"${STAMP}"* 2>/dev/null || true
echo "登记：更新 **EXPERIMENT_REGISTRY** 行 **A-stage2-wikitext-grid-v1**（本跑 **TAG=$TAG** **STAMP=$STAMP**）。"
