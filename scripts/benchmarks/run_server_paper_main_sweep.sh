#!/usr/bin/env bash
# 同机「主文级」扫参：全程固定 WARMUP / REPS，便于论文主图与表格（避免跨机混用不同计时设定）。
# 预期环境：Linux / AutoDL，已激活 conda mamba2，torch CUDA OK；**已装 mamba_ssm + causal-conv1d** 时 Mamba2 走融合路径（低峰值）。
#
# **同机 HF naive 对照**（与 fused 相同网格、建议相同 WARMUP/REPS、可用不同 TAG）：
#   ./scripts/benchmarks/run_server_paper_main_sweep_naive.sh
# 需在无 mamba-ssm / causal-conv1d 的 conda 环境中运行，见该脚本与 SERVER_SWEEP_RUNBOOK §2c。
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

echo "=== paper main sweep (fused 预期)  tag=$TAG  warmup=$WARMUP  reps=$REPS  out=$OUT ==="

MANIFEST="$OUT/paper_main_manifest_${TAG}.txt"
{
  echo "mode=fused_expected"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  python -c "import importlib.util; print('causal_conv1d', bool(importlib.util.find_spec('causal_conv1d')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

# shellcheck source=scripts/benchmarks/_paper_main_sweep_body.sh
source "$ROOT/scripts/benchmarks/_paper_main_sweep_body.sh"

echo "=== 完成 ==="
ls -la "$OUT/paper_main_"*"${TAG}"*
