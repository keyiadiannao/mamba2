# 周历与里程碑（滚动更新）

> 使用方式：每周五勾选本周项，并写下「阻塞项」一行；周一补下周三条**可执行**任务（含负责机器：本地 / AutoDL）。

---

## 当前周期（请每次更新日期范围）

**周期**：2026-04-07 — 2026-04-13

| 状态 | 任务 | 机器 | 产出 |
|------|------|------|------|
| ☑ | 锁定 Python/CUDA 版本，生成 `environment/requirements-mamba2-lock.txt` | 本地 | 已导出 pip freeze |
| ☐ | 仓库克隆到 AutoDL，配置 `data/`、`checkpoints/` 为数据盘路径 | AutoDL | `SYNC` 文档中路径已填 |
| ☑ | Smoke：`scripts/smoke_local.py` | 5060 | registry `X-20260407-smoke-local` |
| ☑ | 阶段 1 入口：玩具树 + Reader 对比脚本骨架 | 本地 | `experiments/A-20260407-toy-tree-reader-bench`、`scripts/benchmark_tree_walk.py` |

**本周阻塞**：_

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

## 历史记录（倒序，每条一行）

| 日期 | 完成项 |
|------|--------|
| 2026-04-07 | 阶段 1 玩具树 + Transformer/GRU 微基准脚本；`requirements-mamba2-lock.txt` |
| 2026-04-07 | `PHASE1_VALIDATION_PLAN.md`；`sweep_tree_benchmark.py`；本地 preset 扫参 CSV |
| 2026-04-07 | `PROJECT_MASTER_PLAN.md`、`CURRENT_SPRINT.md`；文本形树 `benchmark_text_tree.py`；扫参 CSV 增列与 `merge_sweep_csv.py` |
| 2026-04-07 | `data/raw/sample` 联调语料；`prepare_leaves_from_corpus.py`；`AUTODL_SETUP.md` |
| 2026-04-07 | `smoke_mamba_minimal.py`：HF `MambaModel` tiny，无 mamba-ssm |
