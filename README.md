# Mamba × 树状 RAG × 检索头 — 研究仓库

本地开发与小规模验证（RTX 5060 8GB），训练与主实验在 AutoDL（48GB）。**代码与配置以 Git 为唯一真相源；大文件走网盘/面板同步，不进仓库。**

- 全局目标、分类与阶段计划：`docs/PROJECT_OVERVIEW.md`
- 双机分工与上传下载规范：`docs/SYNC_AND_ENVIRONMENTS.md`
- 实验登记（随做随记）：`docs/EXPERIMENT_REGISTRY.md`
- 阶段 1 验证规划与扫参说明：`docs/PHASE1_VALIDATION_PLAN.md`

## 本机 Conda 环境 `mamba2`（推荐）

专用于本仓库；创建与 **RTX 5060（sm_120）需 cu128 版 PyTorch** 的说明见 `environment/MAMBA2.md`。

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2
python scripts\smoke_local.py
```

未激活时可直接使用：`C:\Users\26433\miniconda3\envs\mamba2\python.exe scripts\smoke_local.py`

**不要用** base 里自带的 `+cpu` torch 跑 GPU；5060 也不要装 `cu126` 稳定包（会缺 sm_120 内核），请用 `cu128` 轮子（见 `environment/MAMBA2.md`）。

## 阶段 1 入口（玩具树 + Reader 微基准）

在激活 `mamba2` 且位于仓库根目录时：

```powershell
python scripts\benchmark_tree_walk.py --depth 6 --fanout 2
```

说明与登记见 `experiments/A-20260407-toy-tree-reader-bench/README.md`（GRU 为 Mamba 占位，后续替换）。

### 扫参（CSV）

```powershell
python scripts\sweep_tree_benchmark.py --preset local --out-csv results\metrics\sweep_tree_reader_local.csv
```

自定义网格见 `docs/PHASE1_VALIDATION_PLAN.md`（`--depths`、`--chunk-lens`、`--max-leaves` 等）。
