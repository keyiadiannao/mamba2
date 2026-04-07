# 实验：本地最小 Mamba（X）

- **registry id**: `X-20260409-mamba-minimal-smoke`
- **目的**：在 **8GB** 本机验证 **`Mamba2Model`（默认）** 或 `MambaModel`（`--arch mamba`）前向/反向与显存峰值；**不依赖** `mamba-ssm`（Windows 上常难装）。
- **模型选择**：随机初始化的 **tiny** 配置（默认 2 层、`hidden=256`、`state_size=16`、`expand=2`、`8×64` heads，约 **9.1M 参数**），与预训练大模型无关，仅作**工程冒烟**。

## 命令

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2
python scripts\smoke_mamba_minimal.py
python scripts\smoke_mamba_minimal.py --seq 256 --backward
```

## 说明

- 未安装 `mamba-ssm` 时，Transformers 使用 **顺序 PyTorch 实现**，控制台可能出现一次「fast path is not available」——属预期。
- 需要 **更快或更大** 实验时：在 **Linux / AutoDL** 上安装 `mamba-ssm` 与 `causal-conv1d` 后再跑同一脚本或换官方 checkpoint。
