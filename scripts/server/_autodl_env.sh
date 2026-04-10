# shellcheck shell=bash
# 由同目录下其它脚本 **source**（勿直接执行）。
# 作用：cd 仓库根、激活 conda mamba2、HF 镜像、结果目录。
#
# 可选环境变量：
#   MAMBA2_REPO_ROOT   默认：本文件上两级目录（仓库根）
#   MAMBA2_RESULTS_ROOT 默认：/root/autodl-tmp/mamba2_results
#   HF_ENDPOINT        默认：https://hf-mirror.com
#   CONDA_SH           若自动探测失败，设为 conda.sh 的绝对路径

_ae_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAMBA2_REPO_ROOT="${MAMBA2_REPO_ROOT:-$(cd "$_ae_dir/../.." && pwd)}"
cd "$MAMBA2_REPO_ROOT" || {
  echo "ERROR: cannot cd MAMBA2_REPO_ROOT=$MAMBA2_REPO_ROOT" >&2
  return 1 2>/dev/null || exit 1
}

if [[ "${CONDA_DEFAULT_ENV:-}" != "mamba2" ]]; then
  _conda_sh="${CONDA_SH:-}"
  if [[ -z "$_conda_sh" ]]; then
    if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
      _conda_sh=/root/miniconda3/etc/profile.d/conda.sh
    elif [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
      _conda_sh=/root/anaconda3/etc/profile.d/conda.sh
    fi
  fi
  if [[ -z "$_conda_sh" || ! -f "$_conda_sh" ]]; then
    echo "ERROR: conda.sh not found. Set CONDA_SH=/path/to/conda.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck source=/dev/null
  source "$_conda_sh"
  conda activate mamba2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MAMBA2_RESULTS_ROOT="${MAMBA2_RESULTS_ROOT:-/root/autodl-tmp/mamba2_results}"
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result" "$MAMBA2_RESULTS_ROOT/metrics"
