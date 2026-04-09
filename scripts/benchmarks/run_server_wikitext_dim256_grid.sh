#!/usr/bin/env bash
# Wikitext 浅树 **path-batch**，**dim=256**（其余与 **A2-S2** 同拓扑：{8,16}×{8,12}，fused）。
# 与 **dim=128** 的 **A-stage2-wikitext-grid-v1** **分列登记**（新 **TAG** / **EXPERIMENT_REGISTRY** 行）。
#
#   export TAG=stage2_dim256
#   ./scripts/benchmarks/run_server_wikitext_dim256_grid.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

WARMUP="${WARMUP:-2}"
REPS="${REPS:-8}"
TAG="${TAG:-stage2_dim256}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
GRID="${GRID:-full}"
DIM="${DIM:-256}"

BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics_result"
mkdir -p "$OUT"

MANIFEST="$OUT/benchmark_wikitext_${TAG}_manifest_${STAMP}.txt"
{
  echo "kind=wikitext_fused_dim256_grid"
  echo "dim=$DIM"
  echo "grid=$GRID"
  echo "tag=$TAG"
  echo "stamp=$STAMP"
  echo "warmup=$WARMUP reps=$REPS"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "HF_ENDPOINT=${HF_ENDPOINT:-}"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

run_cell() {
  local n=$1 c=$2
  local json_path="$OUT/benchmark_wikitext_${TAG}_${STAMP}_n${n}_c${c}.json"
  echo "=== dim=$DIM num_leaves=$n chunk_len=$c -> $json_path ==="
  python scripts/benchmarks/benchmark_wikitext_tree.py \
    --num-leaves "$n" --chunk-len "$c" --fanout 2 --dim "$DIM" \
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
echo "登记：在 **EXPERIMENT_REGISTRY** 新增一行（建议 id **A-stage2-wikitext-dim256-v1**），写 **TAG=$TAG** **STAMP=$STAMP** **dim=$DIM**。"
