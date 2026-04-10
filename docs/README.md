# 文档索引（主表）

对外入口：仓库根目录 **`README.md`**。本文件按**职责**分层，减少在 `overview/` 多份总览之间的重复阅读。

## 单一权威（避免双写）

以下主题**只在该处维护长文/长表**；其他文档仅**链接**，不复制同一段落。

| 主题 | 唯一权威 |
|------|----------|
| 叙事框架、基线对照、A/B/C/X 四条线、月级阶段表 | [`overview/PROJECT_MASTER_PLAN.md`](overview/PROJECT_MASTER_PLAN.md) |
| **现状**大表、五轴防混读、§3.5 证据梯、决策 §4、公平性 §6 | [`overview/RESEARCH_STATUS_AND_DIRECTION.md`](overview/RESEARCH_STATUS_AND_DIRECTION.md) |
| 轨道里程碑（A2-S*、B-S*）、**当前收口清单**、P0–P3 / 无云端 §A–§C | [`overview/NEXT_RESEARCH_PLAN.md`](overview/NEXT_RESEARCH_PLAN.md) |
| **本周**勾选、阻塞、周期说明 | [`overview/CURRENT_SPRINT.md`](overview/CURRENT_SPRINT.md) |
| 本机 5060 **可复制命令**与章节索引 | [`environment/LOCAL_5060_RUNBOOK.md`](environment/LOCAL_5060_RUNBOOK.md) |
| 云端扫参 shell、§7 depth、**TAG/STAMP** 约定 | [`environment/SERVER_SWEEP_RUNBOOK.md`](environment/SERVER_SWEEP_RUNBOOK.md) + [`environment/NEXT_EXPERIMENTS_COMMANDS.md`](environment/NEXT_EXPERIMENTS_COMMANDS.md) |
| 实验 id、路径、结论一行 | [`experiments/EXPERIMENT_REGISTRY.md`](experiments/EXPERIMENT_REGISTRY.md) |
| 周历**模板**、历史一行记录 | [`overview/ROADMAP.md`](overview/ROADMAP.md)（可执行项见 **CURRENT_SPRINT**） |

## 0. 三分钟：该读哪份？

| 目的 | 文件 |
|------|------|
| 现状、证据、推荐推进顺序 | [`overview/RESEARCH_STATUS_AND_DIRECTION.md`](overview/RESEARCH_STATUS_AND_DIRECTION.md) |
| 本周在做什么、阻塞项 | [`overview/CURRENT_SPRINT.md`](overview/CURRENT_SPRINT.md) |
| 任务拆条（轨道 A/B、阶段 2、检索头） | [`overview/NEXT_RESEARCH_PLAN.md`](overview/NEXT_RESEARCH_PLAN.md) |
| 投稿用草稿块与检查清单 | [`overview/SUBMISSION_PACK.md`](overview/SUBMISSION_PACK.md) |
| 实验登记与指标文件名 | [`experiments/EXPERIMENT_REGISTRY.md`](experiments/EXPERIMENT_REGISTRY.md) |

## 1. 成文与图表

| 文件 | 说明 |
|------|------|
| [`experiments/PHASE1_MANUSCRIPT.md`](experiments/PHASE1_MANUSCRIPT.md) | 阶段 1 主成稿（含 §8–§10 及阶段 2 并入叙述） |
| [`experiments/PHASE2_DRAFT.md`](experiments/PHASE2_DRAFT.md) | 阶段 2 **表数字、修订日志**；叙事主干以 MANUSCRIPT 为准 |
| [`experiments/FIGURE_CAPTIONS_STAGE1.md`](experiments/FIGURE_CAPTIONS_STAGE1.md) | 主图图注与五轴护栏 |
| [`overview/SUBMISSION_PACK.md`](overview/SUBMISSION_PACK.md) | 投稿结构化草案（A1–A7 等） |

**存档 / 补充（不替代上表主链）**

| 文件 | 说明 |
|------|------|
| [`experiments/PHASE1_COMPLETE_SUMMARY.md`](experiments/PHASE1_COMPLETE_SUMMARY.md) | 阶段 1 收口存档、§7 复跑指针 |
| [`experiments/PHASE1_VALIDATION_PLAN.md`](experiments/PHASE1_VALIDATION_PLAN.md) | 早期验证规划与扫参约定 |

## 2. 规划与总览（`overview/`）

| 文件 | 职责 |
|------|------|
| [`RESEARCH_STATUS_AND_DIRECTION.md`](overview/RESEARCH_STATUS_AND_DIRECTION.md) | **唯一**长篇：现状 + 方向 + 证据梯 |
| [`CURRENT_SPRINT.md`](overview/CURRENT_SPRINT.md) | **当前迭代**勾选与阻塞（周更） |
| [`NEXT_RESEARCH_PLAN.md`](overview/NEXT_RESEARCH_PLAN.md) | 中长期任务表、收口清单 |
| [`PROJECT_MASTER_PLAN.md`](overview/PROJECT_MASTER_PLAN.md) | 里程碑式总体规划（月级） |
| [`PROJECT_OVERVIEW.md`](overview/PROJECT_OVERVIEW.md) | 目标、**A/B/C/X** 分类、**仓库目录树**；执行细节指向上表与 registry |
| [`ROADMAP.md`](overview/ROADMAP.md) | 周历**模板**、里程碑对照与**历史**一行；**阶段 2 详表**见 **NEXT_RESEARCH_PLAN §2**；**可执行勾选**以 **CURRENT_SPRINT** 为准 |

## 3. 实验与数据

- [`experiments/EXPERIMENT_REGISTRY.md`](experiments/EXPERIMENT_REGISTRY.md)
- [`experiments/DATASETS.md`](experiments/DATASETS.md)

## 4. 环境与运行（`environment/`）

- **本机 RTX 5060**：优先 [`LOCAL_5060_RUNBOOK.md`](environment/LOCAL_5060_RUNBOOK.md)
- **云端 / 命令模板**：[`NEXT_EXPERIMENTS_COMMANDS.md`](environment/NEXT_EXPERIMENTS_COMMANDS.md)、[`SERVER_SWEEP_RUNBOOK.md`](environment/SERVER_SWEEP_RUNBOOK.md)、[`SYNC_AND_ENVIRONMENTS.md`](environment/SYNC_AND_ENVIRONMENTS.md)、[`AUTODL_SETUP.md`](environment/AUTODL_SETUP.md)
- **安装与排障**：[`MAMBA_SSM_INSTALL_LINUX.md`](environment/MAMBA_SSM_INSTALL_LINUX.md)、[`SH_CRLF_LINUX.md`](environment/SH_CRLF_LINUX.md)、[`GIT_SERVER_MERGE_UNTRACKED.md`](environment/GIT_SERVER_MERGE_UNTRACKED.md)
- **§7 玩具 JSON（按需）**：[`RUN_AUTOADL_SECTION7_NOW.md`](environment/RUN_AUTOADL_SECTION7_NOW.md)

## 5. 研究笔记（`research/`）

- [`RESEARCH_NOTES.md`](research/RESEARCH_NOTES.md) — SSGS、快照、§7 等
- [`RETRIEVAL_HEAD_NOTES.md`](research/RETRIEVAL_HEAD_NOTES.md) — B 线、检索头叙事与 **B-S2** 边界

---

**子目录一览**

| 目录 | 内容 |
|------|------|
| `docs/overview/` | 规划、sprint、研究状态、投稿包 |
| `docs/experiments/` | 登记、成文稿、阶段 2 草稿、图注、数据说明 |
| `docs/environment/` | 同步、AutoDL、跑书、排障 |
| `docs/research/` | 主题笔记 |
