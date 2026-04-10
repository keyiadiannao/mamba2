# 实验：本地小规模 smoke（X）

- **registry id**: `X-20260407-smoke-local`
- **machine**: 5060 8G（或 CPU）
- **hypothesis**: 本机 PyTorch/CUDA 可用，小 batch 前向无 OOM。

## 命令

在仓库根目录：

```powershell
conda activate mamba2
python scripts\smoke_local.py
```

或：`C:\Users\26433\miniconda3\envs\mamba2\python.exe scripts\smoke_local.py`

可选：先设置数据/输出根目录（见 `docs/environment/runbooks/SYNC_AND_ENVIRONMENTS.md`）：

```powershell
$env:MAMBA2_DATA_ROOT = "D:\cursor_try\mamba2_data"
$env:MAMBA2_RESULTS_ROOT = "D:\cursor_try\mamba2_results"
python scripts/smoke/smoke_local.py
```

## 结果

- 将 `elapsed_s`、`cuda.is_available`、显存信息记入 `docs/experiments/planning/EXPERIMENT_REGISTRY.md` 对应行。
