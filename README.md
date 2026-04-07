# Mamba × 树状 RAG × 检索头 — 研究仓库

本地开发与小规模验证（RTX 5060 8GB），训练与主实验在 AutoDL（48GB）。**代码与配置以 Git 为唯一真相源；大文件走网盘/面板同步，不进仓库。**

- 全局目标、分类与阶段计划：`docs/PROJECT_OVERVIEW.md`
- 双机分工与上传下载规范：`docs/SYNC_AND_ENVIRONMENTS.md`
- 实验登记（随做随记）：`docs/EXPERIMENT_REGISTRY.md`
- 阶段 1 验证规划与扫参说明：`docs/PHASE1_VALIDATION_PLAN.md`
- **总体规划与当前迭代**：`docs/PROJECT_MASTER_PLAN.md`、`docs/CURRENT_SPRINT.md`
- 数据与样例：`docs/DATASETS.md`
- AutoDL 环境步骤：`docs/AUTODL_SETUP.md`

## 本机 Conda 环境 `mamba2`（推荐）

专用于本仓库；创建与 **RTX 5060（sm_120）需 cu128 版 PyTorch** 的说明见 `environment/MAMBA2.md`。

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2
python scripts\smoke_local.py
```

未激活时可直接使用：`C:\Users\26433\miniconda3\envs\mamba2\python.exe scripts\smoke_local.py`

**不要用** base 里自带的 `+cpu` torch 跑 GPU；5060 也不要装 `cu126` 稳定包（会缺 sm_120 内核），请用 `cu128` 轮子（见 `environment/MAMBA2.md`）。

### 本地最小 Mamba（Transformers，8G 友好）

无需 `mamba-ssm` 即可在 5060 上跑通小配置 **`Mamba2Model`**（默认；无融合核时为 naive 回退，适合做 smoke）：

```powershell
python -m pip install "transformers>=4.45" accelerate
python scripts\smoke_mamba_minimal.py
python scripts\smoke_mamba_minimal.py --arch mamba
```

详见 `environment/MAMBA2.md` 与 `experiments/X-20260409-mamba-minimal-smoke/README.md`。

## 阶段 1 入口（玩具树 + Reader 微基准）

在激活 `mamba2` 且位于仓库根目录时：

```powershell
python scripts\benchmark_tree_walk.py --depth 6 --fanout 2
```

说明与登记见 `experiments/A-20260407-toy-tree-reader-bench/README.md`。默认含 **Transformer + GRU + Mamba2** 三路对比；`--no-mamba2` 可关掉 Mamba2。

### 扫参（CSV）

```powershell
python scripts\sweep_tree_benchmark.py --preset local --out-csv results\metrics\sweep_tree_reader_local.csv
```

自定义网格见 `docs/PHASE1_VALIDATION_PLAN.md`（`--depths`、`--chunk-lens`、`--max-leaves` 等）。

多机 CSV 合并：

```powershell
python scripts\merge_sweep_csv.py results\metrics\merged.csv results\metrics\run_a.csv results\metrics\run_b.csv
```

### 文本形浅树（样例叶节点）

```powershell
python scripts\benchmark_text_tree.py --leaf-file experiments\A-20260408-text-shaped-tree\leaves_sample.txt
```

见 `experiments/A-20260408-text-shaped-tree/README.md`。
