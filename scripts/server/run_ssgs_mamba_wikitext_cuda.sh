#!/usr/bin/env bash
# 主线：Wikitext 浅树 + Mamba + SSGS（dfs_ssgs_mamba，与 benchmark_wikitext_tree 同建树）。
# 依赖：已安装 transformers + torch(CUDA)；须先跑 bootstrap_autodl.sh 或本脚本会 source 同环境。
#
# 用法：
#   bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
#
# 额外叶数（空格分隔，与 grid 对齐的 n8/n16/n32/...）：
#   EXTRA_LEAVES="16 32" bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
#
# 同时跑 path-batch 单格（n8 c8 dim128，写 JSON）：
#   RUN_WIKITEXT_SMOKE=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
#
# 合并 CSV 时保留已有 ssgs_mamba_wikitext_grid.csv 中其它行：
#   AGGREGATE_APPEND=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_autodl_env.sh"

STAMP=$(date -u +%Y%m%dT%H%MZ)
OUT_DIR="$MAMBA2_RESULTS_ROOT/metrics_result"
mkdir -p "$OUT_DIR"

if [[ "${RUN_WIKITEXT_SMOKE:-0}" == "1" ]]; then
  echo "== path-batch smoke: benchmark_wikitext_tree n8 c8 dim128 =="
  python scripts/benchmarks/benchmark_wikitext_tree.py \
    --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 \
    --warmup 2 --reps 8 \
    --out-json "$OUT_DIR/benchmark_wikitext_ssgs_bundle_${STAMP}_n8_c8.json"
fi

echo "== SSGS × Mamba × Wikitext：n8 c8 dim128 CUDA（对齐 grid）=="
python scripts/research/demo_ssgs_mamba_wikitext.py \
  --device cuda \
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 --layers 2 \
  --target-leaf-index -1 \
  --out-json "$OUT_DIR/ssgs_mamba_wikitext_n8_c8_dim128_cuda_${STAMP}.json"

for N in ${EXTRA_LEAVES:-}; do
  echo "== SSGS × Mamba × Wikitext：n${N} c8 dim128 CUDA =="
  python scripts/research/demo_ssgs_mamba_wikitext.py \
    --device cuda \
    --num-leaves "$N" --fanout 2 --chunk-len 8 --dim 128 --layers 2 \
    --target-leaf-index -1 \
    --out-json "$OUT_DIR/ssgs_mamba_wikitext_n${N}_c8_dim128_cuda_${STAMP}.json"
done

AGG=(python scripts/research/aggregate_ssgs_mamba_wikitext_json.py \
  -g "$OUT_DIR/ssgs_mamba_wikitext_*.json" \
  --out-csv "$OUT_DIR/ssgs_mamba_wikitext_grid.csv")
if [[ "${AGGREGATE_APPEND:-0}" == "1" ]]; then
  AGG+=(--append)
fi
echo "== 汇总 CSV：${AGG[*]} =="
"${AGG[@]}"

echo "== 完成。请检查 JSON 内 git_sha 与 git rev-parse --short HEAD 一致；登记 EXPERIMENT_REGISTRY（X-20260407-ssgs-mamba-wikitext-tree 或新行）=="
