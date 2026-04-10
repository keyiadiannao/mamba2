# Mamba × 树状 RAG × 检索头 — 研究仓库

本地开发与小规模验证（RTX 5060 8GB），训练与主实验在 AutoDL（48GB）。**代码与配置以 Git 为唯一真相源；大文件走网盘/面板同步，不进仓库。**

**文档**：分层索引 **`docs/README.md`**（先读谁、成文、规划、环境）。**脚本**：`scripts/README.md`。

- **研究现状与方向**（建议先读）：`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`
- **当前迭代**：`docs/overview/execution/CURRENT_SPRINT.md`；**任务展开**：`docs/overview/execution/NEXT_RESEARCH_PLAN.md`
- 全局目标、A/B/C/X 与仓库树：`docs/overview/planning/PROJECT_OVERVIEW.md`；月级规划：`docs/overview/planning/PROJECT_MASTER_PLAN.md`
- 实验登记：`docs/experiments/planning/EXPERIMENT_REGISTRY.md`；阶段 1 成稿：`docs/experiments/phases/PHASE1_MANUSCRIPT.md`；阶段 2 表数字草稿：`docs/experiments/phases/PHASE2_DRAFT.md`；投稿包：`docs/overview/execution/SUBMISSION_PACK.md`
- **指标归档**：`results/metrics_result/`；数据说明：`docs/experiments/planning/DATASETS.md`；早期验证规划：`docs/experiments/planning/PHASE1_VALIDATION_PLAN.md`
- 双机与 AutoDL：`docs/environment/runbooks/SYNC_AND_ENVIRONMENTS.md`、`docs/environment/runbooks/AUTODL_SETUP.md`

## 本机 Conda 环境 `mamba2`（推荐）

专用于本仓库；创建与 **RTX 5060（sm_120）需 cu128 版 PyTorch** 的说明见 `environment/MAMBA2.md`。

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2
python scripts\smoke\smoke_local.py
```

未激活时可直接使用（路径请按本机 conda 安装位置修改）：`…\miniconda3\envs\mamba2\python.exe scripts\smoke\smoke_local.py`（示例见 `environment/MAMBA2.md`）

**不要用** base 里自带的 `+cpu` torch 跑 GPU；5060 也不要装 `cu126` 稳定包（会缺 sm_120 内核），请用 `cu128` 轮子（见 `environment/MAMBA2.md`）。

### 本地最小 Mamba（Transformers，8G 友好）

无需 `mamba-ssm` 即可在 5060 上跑通小配置 **`Mamba2Model`**（默认；无融合核时为 naive 回退，适合做 smoke）：

```powershell
python -m pip install "transformers>=4.45" accelerate
python scripts\smoke\smoke_mamba_minimal.py
python scripts\smoke\smoke_mamba_minimal.py --arch mamba
```

详见 `environment/MAMBA2.md` 与 `experiments/X-20260409-mamba-minimal-smoke/README.md`。

## 阶段 1 入口（玩具树 + Reader 微基准）

在激活 `mamba2` 且位于仓库根目录时：

```powershell
python scripts\benchmarks\benchmark_tree_walk.py --depth 6 --fanout 2
```

说明与登记见 `experiments/A-20260407-toy-tree-reader-bench/README.md`。默认含 **Transformer + GRU + Mamba2** 三路对比；`--no-mamba2` 可关掉 Mamba2。

### 扫参（CSV）

```powershell
python scripts\benchmarks\sweep_tree_benchmark.py --preset local --out-csv results\metrics\sweep_tree_reader_local.csv
```

自定义网格见 `docs/experiments/planning/PHASE1_VALIDATION_PLAN.md`（`--depths`、`--chunk-lens`、`--max-leaves` 等）。

多机 CSV 合并：

```powershell
python scripts\benchmarks\merge_sweep_csv.py results\metrics\merged.csv results\metrics\run_a.csv results\metrics\run_b.csv
```

### 文本形浅树（样例叶节点）

```powershell
python scripts\benchmarks\benchmark_text_tree.py --leaf-file experiments\A-20260408-text-shaped-tree\leaves_sample.txt
```

见 `experiments/A-20260408-text-shaped-tree/README.md`。

### 真因果 LM × 文本形树（最小闭环）

路径上节点 `text` 拼成文档 → `AutoModelForCausalLM` 的 **CE（teacher forcing）** + **续写**；可选 `--train-one-step`。默认小模型 `sshleifer/tiny-gpt2`（需能访问 Hub 或镜像，见 `AUTODL_SETUP` / `HF_ENDPOINT`）；也可用 `--model <本地权重目录>`。

```powershell
python scripts\research\demo_tree_lm_minimal.py --cpu
```

说明与边界见 `docs/research/RESEARCH_NOTES.md` §7.4（**非** path-batch reader 基准，**非** 可学习树上导航）。

**启发式导航指标**（子节点 CE argmin，与金路径对比）：

```powershell
python scripts\research\demo_tree_lm_nav_greedy.py --cpu --eval-all-leaves --out-json results\metrics\tree_lm_nav_greedy_default8_cpu.json
python scripts\research\demo_tree_lm_nav_greedy.py --eval-all-leaves --out-json results\metrics\tree_lm_nav_greedy_default8_cuda.json
```

登记 **X-20260423-tree-lm-nav-greedy**（`tree_lm_nav_greedy_default8_{cpu,cuda}.json` 指标一致）。终端若出现 **`UNEXPECTED` / `loss_type=None`** 来自 Transformers 加载 **tiny-gpt2**，可忽略。**非**训练好的策略。

**目标叶条件、可学习子指针**（与上行对照：需指定 **哪片叶** 为导航目标，**非**盲 CE-argmin）：

```powershell
python scripts\research\demo_tree_lm_nav_learned.py --cpu --epochs 250 --eval-all-leaves --out-json results\metrics\tree_lm_nav_learned_default8_cpu.json
python scripts\research\demo_tree_lm_nav_learned.py --epochs 250 --eval-all-leaves --out-json results\metrics\tree_lm_nav_learned_default8_cuda.json
```

登记 **X-20260424-tree-lm-nav-learned**；默认 8 叶上 **reach_rate** 高于 **X-20260423**；**CPU/CUDA** 指标一致（见 `EXPERIMENT_REGISTRY` 与 `RESEARCH_NOTES` §7.4）。

**SSGS 与 LM 导航并列**（同树、不同任务；**SSGS 必达**、子头 **可能未达**）：

```powershell
python scripts\research\demo_ssgs_lm_nav_compare.py --cpu --epochs 250 --out-json results\metrics\ssgs_lm_nav_compare_default8_cpu.json
```

登记 **X-20260425-ssgs-lm-nav-compare**。叙事边界见 `docs\experiments\FIGURE_CAPTIONS_STAGE1.md` 篇首。

## 单测

**无 PyTorch 依赖**（叶对几何，A2-S3）：`src/rag_tree/__init__.py` 对 `tree` 做 **惰性导出**，仅跑下列文件时不应加载 `torch`。

```powershell
py -m pytest tests/test_path_pair_geometry.py -q
```

**其余 `tests/`**（SSGS、LM 导航等）需要 **能正常 `import torch` 的环境**（推荐 **`conda activate mamba2`**）；在错误 CUDA 栈或 base 环境上可能在 DLL 初始化阶段失败。

```powershell
py -m pytest tests -q
```

或（SSGS 草稿）：

```powershell
python -m unittest tests.test_ssgs -v
```

张量快照/恢复微基准（无 LM，默认 CUDA）：

```powershell
python scripts\benchmarks\benchmark_ssgs_tensor_overhead.py --dim 256 --micro-iters 50000
```

阶段 1 叙事与 **naive vs fused** 对比图（详见 `docs/experiments/planning/PHASE1_VALIDATION_PLAN.md` §6）：

```powershell
python scripts\benchmarks\plot_mamba_naive_vs_fused.py --csv-a results\metrics\sweep_tree_reader_20260410_local5060.csv --label-a "5060 HF naive" --csv-b results\metrics\sweep_autodl_fused.csv --label-b "3090 fused" --out results\metrics\figures\mamba_naive_vs_fused_peak.png
```
