#!/usr/bin/env bash
# 同机 **HF naive** 主文扫参：网格与 run_server_paper_main_sweep.sh 完全一致，便于与 fused 结果用 plot_mamba_naive_vs_fused.py 叠画（同 GPU、同 WARMUP/REPS/TAG 仅内核不同）。
#
# HuggingFace Mamba2 在能 import mamba_ssm / causal_conv1d 时会走融合路径，无运行时开关。
# 本脚本要求当前 Python 环境中 **二者均不可 import**（见下「环境」）。
#
# 用法：
#   conda create -n mamba2_naive --clone mamba2   # 或新建同 torch/transformers 的环境
#   conda activate mamba2_naive
#   pip uninstall -y mamba-ssm causal-conv1d
#   python -c "import importlib.util as u; assert u.find_spec('mamba_ssm') is None"
#   cd /path/to/mamba2 && chmod +x scripts/benchmarks/run_server_paper_main_sweep_naive.sh
#   WARMUP=2 REPS=8 TAG=paper_main_naive_v1 ./scripts/benchmarks/run_server_paper_main_sweep_naive.sh
#
# 输出：与 fused 相同文件名模式 paper_main_*_${TAG}.csv；manifest 中 mamba_ssm 应为 False。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

python - <<'PY'
import importlib.util
import sys

bad = []
for name in ("mamba_ssm", "causal_conv1d"):
    if importlib.util.find_spec(name) is not None:
        bad.append(name)
if bad:
    print(
        "paper_main naive: 当前环境仍可调包: %s — 将走融合/混合路径，不是 HF naive。"
        % bad,
        file=sys.stderr,
    )
    print(
        "请使用无 mamba-ssm 与 causal-conv1d 的 conda 环境，见本脚本头部说明与 SERVER_SWEEP_RUNBOOK §2c。",
        file=sys.stderr,
    )
    sys.exit(1)
print("paper_main naive: fused 栈未加载，将使用 HF 回退实现。", flush=True)
PY

WARMUP="${WARMUP:-2}"
REPS="${REPS:-8}"
TAG="${TAG:-paper_main_naive_$(date +%Y%m%d_%H%M)}"
BASE_OUT="${MAMBA2_RESULTS_ROOT:-$ROOT/results}"
OUT="$BASE_OUT/metrics"
mkdir -p "$OUT"

echo "=== paper main sweep (HF NAIVE)  tag=$TAG  warmup=$WARMUP  reps=$REPS  out=$OUT ==="

MANIFEST="$OUT/paper_main_manifest_${TAG}.txt"
{
  echo "mode=hf_naive"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import importlib.util; print('mamba_ssm', bool(importlib.util.find_spec('mamba_ssm')))"
  python -c "import importlib.util; print('causal_conv1d', bool(importlib.util.find_spec('causal_conv1d')))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} | tee "$MANIFEST"

# shellcheck source=scripts/benchmarks/_paper_main_sweep_body.sh
source "$ROOT/scripts/benchmarks/_paper_main_sweep_body.sh"

echo "=== 完成（HF naive）==="
ls -la "$OUT/paper_main_"*"${TAG}"*
