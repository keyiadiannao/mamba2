#!/usr/bin/env bash
# §7 玩具协议 **S1 / S2 / S3 / S4**：在 **多条树深度** 上各写一套 JSON（**单路径**、**与 path-batch 分列**）。
# 默认 **DEPTHS="5 6"**（路径节点数 6、7；对应 **fanout=2** 时 **32 / 64 叶**），用于在已归档 **depth=4**（**X-20260421-***）之外扩展 **KV / clone / restore** 随深度趋势。
#
#   ./scripts/benchmarks/run_server_section7_depth_sweep.sh
#
# 复含 depth=4（与仓内 20260421 归档同参对照）::
#   DEPTHS="4 5 6" ./scripts/benchmarks/run_server_section7_depth_sweep.sh
#
# 输出前缀默认 **section7_depth**（**SECTION7_TAG**）；**不**读取通用 **TAG=**（避免与 **leavescale_xl** 等混名）。
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
# 输出文件名前缀：**勿**复用其它流水线留在 shell 里的 **TAG=**（例：**stage2_leavescale_xl**）。自定义前缀请设 **SECTION7_TAG=my_tag**。
TAG="${SECTION7_TAG:-section7_depth}"
PYTHON="${PYTHON:-python}"
DEPTHS="${DEPTHS:-5 6}"
CHUNK_LEN="${CHUNK_LEN:-8}"
DIM="${DIM:-128}"
DEVICE="${DEVICE:-cuda}"
# 与仓内 §7 行默认一致
KV_WARMUP="${KV_WARMUP:-3}"
KV_REPS="${KV_REPS:-20}"
RESTORE_WARMUP="${RESTORE_WARMUP:-3}"
RESTORE_REPS="${RESTORE_REPS:-20}"

BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics_result"
mkdir -p "$OUT"

MANIFEST="$OUT/${TAG}_manifest_${STAMP}.txt"
{
  echo "kind=section7_s1_s4_depth_sweep"
  echo "tag=$TAG stamp=$STAMP"
  echo "depths=$DEPTHS chunk_len=$CHUNK_LEN dim=$DIM device=$DEVICE"
  echo "kv_warmup=$KV_WARMUP kv_reps=$KV_REPS restore_warmup=$RESTORE_WARMUP restore_reps=$RESTORE_REPS"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

for d in $DEPTHS; do
  echo "======== depth=$d (tree_depth_param; path has d+1 nodes) ========" >&2
  py() { "$PYTHON" "$@" >/dev/null; }

  echo "  -> S1 Mamba2 cache snap (JSON to metrics_result)" >&2
  py scripts/research/benchmark_mamba2_cache_snapshot_segments.py \
    --device "$DEVICE" --depth "$d" --chunk-len "$CHUNK_LEN" --dim "$DIM" \
    --out-json "$OUT/${TAG}_s1_mamba2_snap_d${d}_${STAMP}.json"

  echo "  -> S2 TF-R1" >&2
  py scripts/research/benchmark_tf_r1_path_segments.py \
    --device "$DEVICE" --depth "$d" --chunk-len "$CHUNK_LEN" --dim "$DIM" \
    --out-json "$OUT/${TAG}_s2_tf_r1_d${d}_${STAMP}.json"

  echo "  -> S3 TF-KV" >&2
  py scripts/research/benchmark_tf_kv_path_segments.py \
    --device "$DEVICE" --depth "$d" --chunk-len "$CHUNK_LEN" --dim "$DIM" \
    --warmup "$KV_WARMUP" --reps "$KV_REPS" \
    --out-json "$OUT/${TAG}_s3_tf_kv_d${d}_${STAMP}.json"

  echo "  -> S4 Mamba2 restore" >&2
  py scripts/research/benchmark_mamba2_cache_restore_segments.py \
    --device "$DEVICE" --depth "$d" --chunk-len "$CHUNK_LEN" --dim "$DIM" \
    --snapshot-device same \
    --warmup "$RESTORE_WARMUP" --reps "$RESTORE_REPS" \
    --out-json "$OUT/${TAG}_s4_mamba2_restore_d${d}_${STAMP}.json"
done

echo "=== done ==="
ls -la "$OUT/${TAG}_"*"_${STAMP}.json" 2>/dev/null || true
echo "登记：在 **EXPERIMENT_REGISTRY** 新增 **X-section7-depth-extension-v1**（或自拟），填 **STAMP=$STAMP** 与各 JSON 路径；与 **path-batch** **分列**。"
