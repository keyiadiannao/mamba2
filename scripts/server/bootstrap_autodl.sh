#!/usr/bin/env bash
# AutoDL / Linux：激活 mamba2、HF 镜像、结果目录，并做最小连通性检查。
# 用法（在仓库内或任意目录）：
#   bash scripts/server/bootstrap_autodl.sh
#
# 若仓库不在默认路径，先 export MAMBA2_REPO_ROOT=/你的路径/mamba2

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_autodl_env.sh"

echo "== mamba2 环境 =="
echo "  REPO:     $MAMBA2_REPO_ROOT"
echo "  HEAD:     $(git rev-parse --short HEAD 2>/dev/null || echo '(not a git repo)')"
echo "  RESULTS:  $MAMBA2_RESULTS_ROOT"
echo "  HF:       $HF_ENDPOINT"

if [[ "${SKIP_CRLF_FIX:-0}" != "1" ]] && command -v find >/dev/null && command -v xargs >/dev/null; then
  find scripts -name '*.sh' -print0 2>/dev/null | xargs -0 sed -i 's/\r$//' || true
fi

echo "== Hugging Face / datasets（Wikitext 两行）=="
python -c "from datasets import load_dataset; load_dataset('wikitext','wikitext-2-raw-v1',split='train[:2]'); print('datasets OK')"

echo "== PyTorch / CUDA =="
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available())"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  python -c "import torch; print('device:', torch.cuda.get_device_name(0))"
else
  echo "WARN: CUDA 不可用；SSGS Wikitext CUDA 脚本将回退 CPU（仍可做 smoke）" >&2
fi

echo "== bootstrap 完成。下一步主线 SSGS：=="
echo "  bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh"
echo "可选（path-batch 同树 harness smoke）："
echo "  RUN_WIKITEXT_SMOKE=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh"
echo "Phase M1（SSGS vs TF-KV 三臂，默认 n8/n16/n32）："
echo "  bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh"
