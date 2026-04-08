# 周历与里程碑（滚动更新）

> 使用方式：每周五勾选本周项，并写下「阻塞项」一行；周一补下周三条**可执行**任务（含负责机器：本地 / AutoDL）。

---

## 当前周期（请每次更新日期范围）

**周期（滚动）**：以 **`docs/overview/CURRENT_SPRINT.md`** 为准。**2026-04-09**：**真 LM 支线（X-20260422–25）** 已归档；**迭代焦点回归主线** — 阶段 1 主图/path-batch 与 §7 的 **审计与成文**，支线 **B** 整体延后。

| 状态 | 任务 | 机器 | 产出 |
|------|------|------|------|
| ☑ | 锁定 Python/CUDA 版本，生成 `environment/requirements-mamba2-lock.txt` | 本地 | 已导出 pip freeze |
| ☐ | 仓库克隆到 AutoDL，配置 `data/`、`checkpoints/` 为数据盘路径 | AutoDL | `SYNC` 文档中路径已填 |
| ☑ | Smoke：`scripts/smoke/smoke_local.py` | 5060 | registry `X-20260407-smoke-local` |
| ☑ | 阶段 1 入口：玩具树 + Reader 对比脚本骨架 | 本地 | `experiments/A-20260407-toy-tree-reader-bench`、`scripts/benchmarks/benchmark_tree_walk.py` |

**本周阻塞**：见 **CURRENT_SPRINT**「阻塞项」。

---

## 里程碑对照（来自总览）

| 周（约） | 里程碑 | 完成标志 |
|----------|--------|----------|
| 1–2 | 阶段 0 基建 | 双机同脚本、registry 有环境实验 |
| 3–6 | 阶段 1 验证 | Mamba vs Transformer 曲线 + 表格 |
| 7–10 | 阶段 2 检索头分析 | 探针报告一页 |
| 11–18 | 阶段 3 注入训练 | 消融表 + 主表 |
| 19+ | 阶段 4 扩展 | 可选论文素材 |

---

## 阶段 2 入口（一页，滚动）

**目标**（对齐 `PROJECT_MASTER_PLAN` 阶段 2）：在 **真语料** 上得到 **浅层树** + **与阶段 1 相同的 path reader harness**，并增加 **至少一个任务级指标**（导航或 QA），避免长期停在纯合成网格。

| 项 | 说明 |
|------|------|
| **语料** | 优先扩展 **`benchmark_wikitext_tree.py` / `hf_corpus.wikitext2_leaf_chunks`**（已登记 **A-20260408-wikitext-3090-fused**）；或另增小语料 + `prepare_leaves_from_corpus.py` |
| **建树** | 保持 **平衡 k 叉 / 自底向上** 与现有 `benchmark_text_tree` 接口一致；RAPTOR 式层次聚类 **可选**，以「能进同一 `run_tree_reader_benchmark`」为硬约束 |
| **指标** | 阶段 1：**latency + m2_peak_mib**；阶段 2：**+1** 如路径准确率、浅层 QA EM/F1 或检索命中率（具体指标在开工前写入 **EXPERIMENT_REGISTRY** 新行） |
| **依赖** | 阶段 1 **主图与登记**审计闭环（**`PHASE1_VALIDATION_PLAN.md` §6.5**）；AutoDL **`HF_ENDPOINT`** 与数据盘路径见 **`AUTODL_SETUP.md`** |
| **风险** | 真树叶块长短不一 → 需固定 **padding/截断** 策略并在 registry 写明，避免与合成树混比 |

**下一步可执行动作**：详见 **`NEXT_RESEARCH_PLAN.md`**（轨道 A/B/X、里程碑 **A2-S0…S4**、**B-S1…S3**、建议两周任务）。

---

## 历史记录（倒序，每条一行）

| 日期 | 完成项 |
|------|--------|
| 2026-04-09 | **决策**：真 LM 导航 **X-20260422–25** 登记与 CUDA 对比 JSON 收口；**回归主线**（path-batch + §7）；**B（子头加强）** 延后 — 见 **CURRENT_SPRINT**「决策记录」与「后续研究方向」 |
| 2026-04-09 | **主线执行**：`PHASE1_VALIDATION_PLAN.md` **§6.5** 主文登记↔CSV↔图审计；**ROADMAP** 增加 **阶段 2 入口（一页）** |
| 2026-04-09 | **`NEXT_RESEARCH_PLAN.md`**：阶段 2 / 检索头 B / 成文与 S5 **展开** |
| 2026-04-09 | **`RESEARCH_STATUS_AND_DIRECTION.md`**：现状、四条轴、决策原则与 **推荐执行顺序**（统领 `NEXT_RESEARCH_PLAN`） |
| 2026-04-08 | **§7 串行复跑**：AutoDL `run_path_protocol_cuda.sh` 全量通过；数值与 **§7.3.1** / `*_20260421.json` 同阶（见 **`PHASE1_COMPLETE_SUMMARY` 附录 B**） |
| 2026-04-07 | 阶段 1 玩具树 + Transformer/GRU 微基准脚本；`requirements-mamba2-lock.txt` |
| 2026-04-07 | `docs/experiments/PHASE1_VALIDATION_PLAN.md`；`scripts/benchmarks/sweep_tree_benchmark.py`；本地 preset 扫参 CSV |
| 2026-04-07 | `docs/overview/PROJECT_MASTER_PLAN.md`、`CURRENT_SPRINT.md`；文本形树 `scripts/benchmarks/benchmark_text_tree.py`；扫参 CSV 增列与 `merge_sweep_csv.py` |
| 2026-04-07 | `data/raw/sample` 联调语料；`prepare_leaves_from_corpus.py`；`AUTODL_SETUP.md` |
| 2026-04-07 | `smoke_mamba_minimal.py`：HF `MambaModel` tiny，无 mamba-ssm |
| 2026-04-07 | `Mamba2PathReader` 接入树路径基准（TF/GRU/Mamba2 三对比） |
