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
# 下面一行请换成官网生成的 GPU 版 wheel 命令，例如：
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

python -m pip install pyyaml tqdm "transformers>=4.45" accelerate
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
python scripts/smoke_local.py
python scripts/smoke_mamba_minimal.py
python scripts/sweep_tree_benchmark.py --preset local --out-csv results/metrics/sweep_autodl.csv
```

将 `sweep_autodl.csv` 下载到本机后，可与本地 CSV 合并：

```powershell
python scripts\merge_sweep_csv.py results\metrics\merged_local_autodl.csv results\metrics\sweep_tree_reader_20260407_local.csv results\metrics\sweep_autodl.csv
```

## 5. mamba-ssm（可选）

依赖 CUDA 与 torch 版本；在激活 `mamba2` 后按 [state-spaces/mamba](https://github.com/state-spaces/mamba) 说明安装。成功后在 `EXPERIMENT_REGISTRY.md` 记一条 `import mamba_ssm` + 微型 forward。

## 6. Git 身份与私有仓库

若仓库为私有，需配置 [PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) 或 SSH key；勿将 token 写入仓库。
