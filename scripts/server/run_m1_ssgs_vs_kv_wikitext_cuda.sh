#!/usr/bin/env bash
# Phase **M1**：**SSGS Mamba** vs **TF-KV**（clone + truncate）**同树 DFS**，Wikitext，**CUDA**。
# 与 **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** 一致；结果写入 **`$MAMBA2_RESULTS_ROOT/metrics_result/`**。
#
# 用法：
#   bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
#
# 叶数（空格分隔；**fanout=2** 时须为 **2^depth**，如 8 16 32 64）：
#   M1_LEAVES="8 16 32 64" bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
#
# 仅两臂（**无** `tf_kv_truncate_arm`，`--no-tf-kv-truncate`）：
#   M1_NO_TRUNCATE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
#
# 固定 **STAMP**（UTC）：
#   M1_STAMP=20260407T1200Z bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
#
# **L3 隐状态**（每叶数 JSON 变大；**n8 单点** 常用）：
#   M1_WITH_L3=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
#
# **L3 下游 CE**（固定随机叶头；与树 LM「可学习 vs CE」**不同 harness**，见脚本 --help / M1 文档）：
#   M1_WITH_L3_DOWNSTREAM_CE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
#
# **汇总 CSV**（默认跑）：**`ssgs_vs_kv_tree_nav_wikitext_*.json`** → **`ssgs_vs_kv_wikitext_nav_grid.csv`**  
#   跳过：**`SKIP_M1_AGGREGATE=1`**；与旧表合并：**`AGGREGATE_APPEND=1`**
#
# 依赖：**`scripts/server/_autodl_env.sh`**（conda **mamba2**、**HF_ENDPOINT**、**MAMBA2_RESULTS_ROOT**）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_autodl_env.sh"

STAMP="${M1_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
OUT_DIR="${MAMBA2_RESULTS_ROOT}/metrics_result"
mkdir -p "$OUT_DIR"

EXTRA_FLAGS=()
if [[ "${M1_NO_TRUNCATE:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--no-tf-kv-truncate)
fi
if [[ "${M1_WITH_L3:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--l3-tf-kv-hidden)
fi
if [[ "${M1_WITH_L3_DOWNSTREAM_CE:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--l3-tf-kv-downstream-ce)
fi

SUFFIX="3arm"
if [[ "${M1_NO_TRUNCATE:-0}" == "1" ]]; then
  SUFFIX="2arm"
fi

for N in ${M1_LEAVES:-8 16 32}; do
  echo "== M1 ssgs_vs_kv Wikitext n${N} c8 dim128 CUDA (${SUFFIX}) STAMP=${STAMP} =="
  python scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py \
    --device cuda \
    --num-leaves "$N" --fanout 2 --chunk-len 8 --dim 128 --layers 2 \
    --tf-layers 2 --tf-nhead 8 --ff-mult 4 \
    --target-leaf-index -1 \
    "${EXTRA_FLAGS[@]}" \
    --out-json "$OUT_DIR/ssgs_vs_kv_tree_nav_wikitext_n${N}_cuda_${SUFFIX}_${STAMP}.json"
done

if [[ "${SKIP_M1_AGGREGATE:-0}" != "1" ]]; then
  AGG=(python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py \
    -g "$OUT_DIR/ssgs_vs_kv_tree_nav_wikitext_*.json" \
    --out-csv "$OUT_DIR/ssgs_vs_kv_wikitext_nav_grid.csv")
  if [[ "${AGGREGATE_APPEND:-0}" == "1" ]]; then
    AGG+=(--append)
  fi
  echo "== M1 汇总 CSV：${AGG[*]} =="
  "${AGG[@]}"
fi

echo "== 完成。核对各 JSON 内 **git_sha**；**EXPERIMENT_REGISTRY** 更新 **X-ssgs-vs-kv-tree-nav-m1**；**CSV** 见 **ssgs_vs_kv_wikitext_nav_grid.csv** =="
