# AutoDL 实例：克隆仓库与跑通基准

> 在 **Linux + NVIDIA GPU** 上复现与本仓库一致的 smoke / 扫参；数据与 checkpoint 建议放在数据盘（如 `/root/autodl-tmp`）。

## 1. 系统信息

```bash
nvidia-smi
uname -a
```

根据驱动与 CUDA 可用性，到 [pytorch.org](https://pytorch.org/get-started/locally/) 选择 **Linux + Pip + 对应 CUDA** 的安装命令（常见为 `cu126`；若实例 CUDA 更新可试 `cu128`）。

## 2. 克隆与 Conda

```bash
cd /root/autodl-tmp   # 或你的数据盘挂载点
git clone https://github.com/keyiadiannao/mamba2.git
cd mamba2

# Miniconda 若已预装：
conda create -n mamba2 python=3.11 pip -y
conda activate mamba2

python -m pip install -U pip
# PyTorch 2.11 常与 setuptools<82 配套；装 torch 前建议：
python -m pip install "setuptools>=70,<82"
# 下面一行请换成官网生成的 GPU 版 wheel 命令，例如：
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

python -m pip install pyyaml tqdm "transformers>=4.45" accelerate datasets
```

## 3. 环境变量（建议）

```bash
export MAMBA2_DATA_ROOT=/root/autodl-tmp/mamba2_data
export MAMBA2_CKPT_ROOT=/root/autodl-tmp/mamba2_ckpt
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_DATA_ROOT" "$MAMBA2_CKPT_ROOT" "$MAMBA2_RESULTS_ROOT"
```

## 4. 验证

```bash
cd /root/autodl-tmp/mamba2
python scripts/smoke/smoke_local.py
python scripts/smoke/smoke_mamba_minimal.py
python scripts/benchmarks/sweep_tree_benchmark.py --preset local --out-csv results/metrics/sweep_autodl.csv
```

**服务器空闲后、与本地 naive 网格对齐的批量扫参**：见 **`docs/environment/SERVER_SWEEP_RUNBOOK.md`** 与 `scripts/benchmarks/run_server_sweep_aligned.sh`。

### 本机合并 CSV（路径怎么填？）

第三个参数必须是：**你已经下载到 Windows 上的 `sweep_autodl.csv` 的完整路径**。

- 若用 **PyCharm Deployment / SFTP**：把远程 `mamba2/results/metrics/sweep_autodl.csv` 同步到本机工程，例如  
  `d:\cursor_try\mamba2\results\metrics\sweep_autodl.csv`  
  则合并命令为：

```powershell
cd d:\cursor_try\mamba2
python scripts\benchmarks\merge_sweep_csv.py results\metrics\merged_5060_3090.csv results\metrics\sweep_tree_reader_20260407_local.csv results\metrics\sweep_autodl.csv
```

- 若文件在 **下载文件夹**（示例，按你的用户名改）：

```powershell
python scripts\benchmarks\merge_sweep_csv.py results\metrics\merged_5060_3090.csv results\metrics\sweep_tree_reader_20260407_local.csv C:\Users\26433\Downloads\sweep_autodl.csv
```

**不要用** 带「下载路径」字样的占位符；在资源管理器里对 `sweep_autodl.csv` **右键 → 属性** 复制「位置」+ 文件名，或 PyCharm 里右键文件 **Copy Path/Reference**。

## 5. mamba-ssm（可选，建议 Linux 上做）

**分步说明与收益**见 **`docs/environment/MAMBA_SSM_INSTALL_LINUX.md`**（含 `causal-conv1d` 顺序、验证命令、常见问题）。成功后在 `docs/experiments/EXPERIMENT_REGISTRY.md` 再记一条带 `mamba_ssm` 的扫参或 `benchmark_tree_walk` 对比。

## 6. Git 身份与私有仓库

若仓库为私有，需配置 [PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) 或 SSH key；勿将 token 写入仓库。
