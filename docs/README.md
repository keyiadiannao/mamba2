# 文档索引（主表）

对外入口：仓库根目录 **`README.md`**。目录已按 **总体规划 / 实施**、**执行手册 / 排障** 分层；**单一权威**矩阵避免双写。

## 文件夹结构（速查）

| 路径 | 类别 | 内容 |
|------|------|------|
| **`docs/overview/planning/`** | 总体规划 | 月级叙事、现状大表、项目总览、周历模板 |
| **`docs/overview/execution/`** | 实施与迭代 | 下一步任务展开、当前 sprint 勾选、投稿包 |
| **`docs/experiments/planning/`** | 实验总体规划 | 登记册、数据约定、阶段 1 验证计划 |
| **`docs/experiments/phases/`** | 阶段成稿与素材 | 阶段 1/2 手稿、图注、阶段总结 |
| **`docs/environment/runbooks/`** | 执行手册 | 本机 5060、AutoDL、云端扫参、同步、安装步骤 |
| **`docs/environment/troubleshooting/`** | 问题与解决 | CRLF、`bash\r`、Git 合并与未跟踪冲突等 |
| **`docs/research/`** | 研究笔记 | SSGS、§7、检索头叙事 |

## 单一权威（避免双写）

| 主题 | 唯一权威 |
|------|----------|
| 叙事框架、基线对照、A/B/C/X、月级阶段表、**§1.0 主验证轴/副线** | [`overview/planning/PROJECT_MASTER_PLAN.md`](overview/planning/PROJECT_MASTER_PLAN.md) |
| **现状**大表、七轴、§3.5 证据梯、决策 §4、公平性 §6 | [`overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`](overview/planning/RESEARCH_STATUS_AND_DIRECTION.md) |
| **阶段 0→结题**（每阶段实验 + 成功标准；与 L1–L4 对齐） | [`overview/planning/RESEARCH_PHASES_0_TO_DONE.md`](overview/planning/RESEARCH_PHASES_0_TO_DONE.md) |
| 轨道里程碑、**当前收口清单**、P0–P3、后备推进 § | [`overview/execution/NEXT_RESEARCH_PLAN.md`](overview/execution/NEXT_RESEARCH_PLAN.md) |
| **本周**勾选、阻塞 | [`overview/execution/CURRENT_SPRINT.md`](overview/execution/CURRENT_SPRINT.md) |
| 本机 5060 **可复制命令** | [`environment/runbooks/LOCAL_5060_RUNBOOK.md`](environment/runbooks/LOCAL_5060_RUNBOOK.md) |
| 云端扫参 shell、§7 depth、命令模板 | [`environment/runbooks/SERVER_SWEEP_RUNBOOK.md`](environment/runbooks/SERVER_SWEEP_RUNBOOK.md) + [`environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md`](environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md)（**§12** 阶段 5 聚合/测试） |
| 实验 id、路径、结论一行 | [`experiments/planning/EXPERIMENT_REGISTRY.md`](experiments/planning/EXPERIMENT_REGISTRY.md) |
| **2026-04 服务器批次** JSON / CSV 索引（M1、SSGS、leavescale、B-S2+） | [`experiments/planning/DATA_ARCHIVE_202604_SERVER.md`](experiments/planning/DATA_ARCHIVE_202604_SERVER.md) |
| 周历**模板**与历史 | [`overview/planning/ROADMAP.md`](overview/planning/ROADMAP.md)（勾选以 **CURRENT_SPRINT** 为准） |

## 0. 三分钟：该读哪份？

| 目的 | 文件 |
|------|------|
| 现状、证据、推荐推进顺序 | [`overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`](overview/planning/RESEARCH_STATUS_AND_DIRECTION.md) |
| 本周在做什么、阻塞项 | [`overview/execution/CURRENT_SPRINT.md`](overview/execution/CURRENT_SPRINT.md) |
| 任务拆条、收口清单 | [`overview/execution/NEXT_RESEARCH_PLAN.md`](overview/execution/NEXT_RESEARCH_PLAN.md) |
| **0→结题** 阶段表与阶段 5 勾选 | [`overview/planning/RESEARCH_PHASES_0_TO_DONE.md`](overview/planning/RESEARCH_PHASES_0_TO_DONE.md) |
| **Mamba+树+SSGS** 整合主线（M1）工具与缺口 | [`experiments/planning/SSGS_MAINLINE_M1.md`](experiments/planning/SSGS_MAINLINE_M1.md) |
| 投稿草稿块与检查清单 | [`overview/execution/SUBMISSION_PACK.md`](overview/execution/SUBMISSION_PACK.md) |
| 实验登记 | [`experiments/planning/EXPERIMENT_REGISTRY.md`](experiments/planning/EXPERIMENT_REGISTRY.md) |
| 四月服务器数据归档（路径 + STAMP） | [`experiments/planning/DATA_ARCHIVE_202604_SERVER.md`](experiments/planning/DATA_ARCHIVE_202604_SERVER.md) |

## 1. 成文与阶段素材（`experiments/phases/`）

| 文件 | 说明 |
|------|------|
| [`phases/PHASE1_MANUSCRIPT.md`](experiments/phases/PHASE1_MANUSCRIPT.md) | 阶段 1 主成稿（含 §8–§10） |
| [`phases/PHASE2_DRAFT.md`](experiments/phases/PHASE2_DRAFT.md) | 阶段 2 表数字、修订日志 |
| [`phases/FIGURE_CAPTIONS_STAGE1.md`](experiments/phases/FIGURE_CAPTIONS_STAGE1.md) | 主图图注与七轴护栏（含 **M1**、**L3 轨迹**） |
| [`phases/PHASE1_COMPLETE_SUMMARY.md`](experiments/phases/PHASE1_COMPLETE_SUMMARY.md) | 阶段 1 存档与 §7 复跑指针 |

**实验侧总体规划**：[`planning/EXPERIMENT_REGISTRY.md`](experiments/planning/EXPERIMENT_REGISTRY.md)、[`planning/DATA_ARCHIVE_202604_SERVER.md`](experiments/planning/DATA_ARCHIVE_202604_SERVER.md)、[`planning/DATASETS.md`](experiments/planning/DATASETS.md)、[`planning/PHASE1_VALIDATION_PLAN.md`](experiments/planning/PHASE1_VALIDATION_PLAN.md)、[`planning/SSGS_MAINLINE_M1.md`](experiments/planning/SSGS_MAINLINE_M1.md)（**Phase M1**）。

## 2. Overview 总体规划（`overview/planning/`）

| 文件 | 职责 |
|------|------|
| [`RESEARCH_STATUS_AND_DIRECTION.md`](overview/planning/RESEARCH_STATUS_AND_DIRECTION.md) | 现状 + 方向 + 证据梯 |
| [`RESEARCH_PHASES_0_TO_DONE.md`](overview/planning/RESEARCH_PHASES_0_TO_DONE.md) | 阶段 0–7、实验与成功标准、阶段 5 清单 |
| [`PROJECT_MASTER_PLAN.md`](overview/planning/PROJECT_MASTER_PLAN.md) | 月级规划、四条线、风险 |
| [`PROJECT_OVERVIEW.md`](overview/planning/PROJECT_OVERVIEW.md) | 目标、A/B/C/X、**仓库目录树** |
| [`ROADMAP.md`](overview/planning/ROADMAP.md) | 周历模板与里程碑对照 |

## 3. Overview 实施（`overview/execution/`）

| 文件 | 职责 |
|------|------|
| [`NEXT_RESEARCH_PLAN.md`](overview/execution/NEXT_RESEARCH_PLAN.md) | 任务展开、收口清单、P1–P3 |
| [`CURRENT_SPRINT.md`](overview/execution/CURRENT_SPRINT.md) | 当前迭代勾选 |
| [`SUBMISSION_PACK.md`](overview/execution/SUBMISSION_PACK.md) | 投稿用 A1–A7 等 |

## 4. 环境：执行手册（`environment/runbooks/`）

- **本机 5060**：[`LOCAL_5060_RUNBOOK.md`](environment/runbooks/LOCAL_5060_RUNBOOK.md)
- **云端命令模板**：[`NEXT_EXPERIMENTS_COMMANDS.md`](environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md)（**§12**）、[`SERVER_SWEEP_RUNBOOK.md`](environment/runbooks/SERVER_SWEEP_RUNBOOK.md)
- **同步与实例**：[`SYNC_AND_ENVIRONMENTS.md`](environment/runbooks/SYNC_AND_ENVIRONMENTS.md)、[`AUTODL_SETUP.md`](environment/runbooks/AUTODL_SETUP.md)
- **融合核安装步骤**：[`MAMBA_SSM_INSTALL_LINUX.md`](environment/runbooks/MAMBA_SSM_INSTALL_LINUX.md)
- **§7 depth 一键**：[`RUN_AUTOADL_SECTION7_NOW.md`](environment/runbooks/RUN_AUTOADL_SECTION7_NOW.md)

## 5. 环境：排障（`environment/troubleshooting/`）

- [`SH_CRLF_LINUX.md`](environment/troubleshooting/SH_CRLF_LINUX.md) — `bash\r` / CRLF
- [`GIT_SERVER_MERGE_UNTRACKED.md`](environment/troubleshooting/GIT_SERVER_MERGE_UNTRACKED.md) — 合并与未跟踪文件

## 6. 研究笔记（`research/`）

- [`RESEARCH_NOTES.md`](research/RESEARCH_NOTES.md)
- [`RETRIEVAL_HEAD_NOTES.md`](research/RETRIEVAL_HEAD_NOTES.md)
