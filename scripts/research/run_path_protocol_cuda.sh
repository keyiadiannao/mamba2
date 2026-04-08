#!/usr/bin/env bash
# 一键复跑 §7 玩具路径协议 S1–S4（CUDA）。默认 depth4/chunk8/dim128，与登记 X-20260421-* 对齐。
# 用法（仓库根）：
#   bash scripts/research/run_path_protocol_cuda.sh
# 输出目录：若已设 MAMBA2_RESULTS_ROOT 则写入 $MAMBA2_RESULTS_ROOT/metrics/；否则写入 results/metrics/。
# 须为 LF 换行；若从 Windows 检出后出现 set: pipefail 报错，见 docs/environment/SH_CRLF_LINUX.md
set -eu
set -o pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

if [[ -n "${MAMBA2_RESULTS_ROOT:-}" ]]; then
  METRICS="$MAMBA2_RESULTS_ROOT/metrics"
else
  METRICS="$REPO/results/metrics"
fi
mkdir -p "$METRICS"

STAMP="$(date -u +%Y%m%dT%H%MZ)"
PY="${PYTHON:-python}"

echo "==> METRICS=$METRICS STAMP=$STAMP"

echo "==> S1 Mamba cache clone"
$PY scripts/research/benchmark_mamba2_cache_snapshot_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$METRICS/mamba2_cache_snap_segments_depth4_cuda_${STAMP}.json"

echo "==> S2 TF-R1"
$PY scripts/research/benchmark_tf_r1_path_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$METRICS/tf_r1_path_segments_depth4_cuda_${STAMP}.json"

echo "==> S3 TF-KV (linear path)"
$PY scripts/research/benchmark_tf_kv_path_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$METRICS/tf_kv_path_segments_depth4_cuda_${STAMP}.json"

echo "==> S3 TF-KV branch-truncate demo (optional second JSON)"
$PY scripts/research/benchmark_tf_kv_path_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 --branch-truncate-demo \
  --out-json "$METRICS/tf_kv_path_segments_depth4_cuda_branchdemo_${STAMP}.json"

echo "==> S4 SSM restore (snapshot on GPU)"
$PY scripts/research/benchmark_mamba2_cache_restore_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$METRICS/mamba2_cache_restore_depth4_cuda_same_${STAMP}.json"

echo "==> S4 SSM restore (snapshot on CPU)"
$PY scripts/research/benchmark_mamba2_cache_restore_segments.py --device cuda --snapshot-device cpu \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$METRICS/mamba2_cache_restore_depth4_cuda_fromcpu_${STAMP}.json"

echo "Done. Compare with archived results/metrics/*_20260421.json and RESEARCH_NOTES §7.3.1."
