#!/usr/bin/env bash
# Wikitext 浅树 **path-batch**：固定 **chunk_len** 与 **dim**，扫描 **num_leaves**（须为 **fanout^depth**，默认 **fanout=2**）。
# 用于观察 **m2_peak / per_step** 随 **叶数（路径 batch、路径长度）** 变化；与 **四格网格** **分列登记**（新 **TAG**）。
#
# **注意**：**`TransformerPathReader`** 对每条路径做 **整段 self-attention**（**O(T²)** per path）；随深度增加 **TF** 往往比 **GRU/Mamba2** 涨得更陡（见 **`src/rag_tree/readers.py`**）。
#
#   export TAG=stage2_leavescale
#   ./scripts/benchmarks/run_server_wikitext_leavescale.sh
#
# 省时（跳过 64 叶）::
#   LEAVES="8 16 32" ./scripts/benchmarks/run_server_wikitext_leavescale.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

WARMUP="${WARMUP:-2}"
REPS="${REPS:-8}"
TAG="${TAG:-stage2_leavescale}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
CHUNK_LEN="${CHUNK_LEN:-8}"
DIM="${DIM:-128}"
FANOUT="${FANOUT:-2}"
# 空格分隔；默认 8→d3、16→d4、32→d5、64→d6
LEAVES="${LEAVES:-8 16 32 64}"

BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics_result"
mkdir -p "$OUT"

MANIFEST="$OUT/benchmark_wikitext_${TAG}_manifest_${STAMP}.txt"
{
  echo "kind=wikitext_fused_leavescale"
  echo "tag=$TAG"
  echo "stamp=$STAMP"
  echo "chunk_len=$CHUNK_LEN dim=$DIM fanout=$FANOUT"
  echo "leaves=$LEAVES"
  echo "warmup=$WARMUP reps=$REPS"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "HF_ENDPOINT=${HF_ENDPOINT:-}"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

for n in $LEAVES; do
  json_path="$OUT/benchmark_wikitext_${TAG}_${STAMP}_n${n}_c${CHUNK_LEN}.json"
  echo "=== num_leaves=$n chunk_len=$CHUNK_LEN dim=$DIM -> $json_path ==="
  python scripts/benchmarks/benchmark_wikitext_tree.py \
    --num-leaves "$n" --fanout "$FANOUT" --chunk-len "$CHUNK_LEN" --dim "$DIM" \
    --warmup "$WARMUP" --reps "$REPS" \
    --out-json "$json_path"
done

CSV_OUT="$OUT/benchmark_wikitext_${TAG}_grid_${STAMP}.csv"
echo "=== aggregate -> $CSV_OUT ==="
python scripts/benchmarks/aggregate_wikitext_tree_json_grid.py \
  --base-dir "$BASE_OUT" \
  --glob "metrics_result/benchmark_wikitext_${TAG}_${STAMP}_n*_c*.json" \
  --out-csv "$CSV_OUT"

echo "=== done ==="
ls -la "$OUT/benchmark_wikitext_${TAG}"*"${STAMP}"* 2>/dev/null || true
echo "登记：在 **EXPERIMENT_REGISTRY** 新增 **A-stage2-wikitext-leavescale-v1**，写 **TAG=$TAG** **STAMP=$STAMP**。"
