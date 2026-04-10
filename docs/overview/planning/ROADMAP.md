# 周历与里程碑（滚动更新）

> **维护**：**勾选、阻塞、本周三条**以 **`docs/overview/execution/CURRENT_SPRINT.md`** 为主表。本文件保留**里程碑模板**与历史周结构；避免与 sprint **双写**同一批可执行项。

> 使用方式（模板）：每周五勾选本周项，并写下「阻塞项」一行；周一补下周三条**可执行**任务（含负责机器：本地 / AutoDL）。

---

## 当前周期（请每次更新日期范围）

**周期（滚动）**：以 **`docs/overview/execution/CURRENT_SPRINT.md`** 为准。**2026-04-09**：**真 LM 支线（X-20260422–25）** 已归档；**迭代焦点回归主线** — 阶段 1 主图/path-batch 与 §7 的 **审计与成文**，支线 **B** 整体延后。

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

## 阶段 2 入口

**不在此重复**：目标、语料/建树约束、里程碑 **A2-S0…S4**、**B-S1…S3** 的**完整表**见 **`docs/overview/execution/NEXT_RESEARCH_PLAN.md` §1–§3**；**现状与六轴**见 **`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md` §2–§3**。

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
| 2026-04-07 | `docs/experiments/planning/PHASE1_VALIDATION_PLAN.md`；`scripts/benchmarks/sweep_tree_benchmark.py`；本地 preset 扫参 CSV |
| 2026-04-07 | `docs/overview/planning/PROJECT_MASTER_PLAN.md`、`CURRENT_SPRINT.md`；文本形树 `scripts/benchmarks/benchmark_text_tree.py`；扫参 CSV 增列与 `merge_sweep_csv.py` |
| 2026-04-07 | `data/raw/sample` 联调语料；`prepare_leaves_from_corpus.py`；`AUTODL_SETUP.md` |
| 2026-04-07 | `smoke_mamba_minimal.py`：HF `MambaModel` tiny，无 mamba-ssm |
| 2026-04-07 | `Mamba2PathReader` 接入树路径基准（TF/GRU/Mamba2 三对比） |
