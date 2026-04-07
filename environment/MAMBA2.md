# Conda 环境 `mamba2`（本项目专用）

## 为何用 cu128

**NVIDIA RTX 5060（Laptop）为 sm_120（Blackwell）**。带 **cu126** 的 PyTorch 不包含该架构的 CUDA 内核，会在 GPU 上报 `no kernel image`。请使用 **CUDA 12.8** 对应的官方轮子（`+cu128`）。

## 一次性创建（Windows + Miniconda）

若 `conda create` 提示需接受 Anaconda 渠道条款：

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

创建环境：

```powershell
conda create -n mamba2 python=3.11 pip -y
conda activate mamba2
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

可选（后续脚本常用）：

```powershell
python -m pip install pyyaml tqdm
```

**Mamba-2 最小测试（推荐）**：用 HuggingFace **`Mamba2Model`** 随机初始化小配置（`smoke_mamba_minimal.py` **默认**），**无需**在 Windows 上编译 `mamba-ssm`（会用 naive PyTorch 回退，首次可能打印一次「fast path is not available」）。

```powershell
python -m pip install "transformers>=4.45" accelerate
python scripts\smoke_mamba_minimal.py
python scripts\smoke_mamba_minimal.py --arch mamba
```

第二行：Mamba-2（默认）；第三行：原版 `MambaModel`（对照）。

云端若安装 `mamba-ssm` + `causal-conv1d` 后，同一脚本可走融合内核（更快）。

锁定当前环境依赖（建议在重大变更后重跑）：

```powershell
python -m pip freeze > environment\requirements-mamba2-lock.txt
```

## 日常使用

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2
python scripts\smoke_local.py
```

本机解释器完整路径（未激活 conda 时）：

`C:\Users\26433\miniconda3\envs\mamba2\python.exe`

## 本机用 conda 还是项目里的 `.venv`？

**推荐：Conda 环境名 `mamba2`**（与上文一致）。仓库**没有**自带强制 `.venv`；若你用 `python -m venv .venv` 也可以，但文档与 `requirements-mamba2-lock.txt` 均以 **Miniconda + mamba2** 为准，两台机器对齐时最省事。

## AutoDL 说明

云端 GPU 架构可能与 5060 不同：若在 **A100 / 4090** 等上安装，可选用对应 CUDA 版本的 PyTorch（例如 `cu126`），**不必**强行与本地一致；以 `nvidia-smi` 与 [pytorch.org](https://pytorch.org/get-started/locally/) 为准。
