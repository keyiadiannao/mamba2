# 文档索引（主表）

对外入口：仓库根目录 **`README.md`**。目录已按 **总体规划 / 实施**、**执行手册 / 排障** 分层；**单一权威**矩阵避免双写。**从现在 → L1 端到端实验完成** 的时间序主线：**[`overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md`](overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md)**。

## 文件夹结构（速查）

| 路径 | 类别 | 内容 |
|------|------|------|
| **`docs/overview/planning/`** | 总体规划 | 月级叙事、现状大表、项目总览、周历模板；**`NARRATIVE_MAINLINE_TREE_READER_SSGS`** 主线链与数据索引；**`MOTIVATION_MAMBA_TREE_RAG_SSGS_REFERENCE`** 动机与外部叙事参照；**`RAPTOR_INSPIRED_NEXT_PHASE`** **RAPTOR 式 × SSGS × Mamba** 下一重点（**非**全量复现 RAPTOR） |
| **`docs/overview/execution/`** | 实施与迭代 | 下一步任务展开、当前 sprint 勾选、投稿包 |
| **`docs/overview/engineering/`** | **工程北星**（战略 B） | **Runner / G1–G5 / Sprint**；与 **`experiments/`** 实证线 **文档分列**、**代码复用** |
| **`docs/experiments/planning/`** | 实验总体规划 | 登记册、数据约定、阶段 1 验证计划 |
| **`docs/experiments/phases/`** | 阶段成稿与素材 | 阶段 1/2 手稿、图注、**`FOUNDATION_STAGE_FORMAL_SUMMARY`** 基础阶段正式浓缩稿、阶段总结 |
| **`docs/environment/runbooks/`** | 执行手册 | 本机 5060、AutoDL、云端扫参、同步、安装步骤 |
| **`docs/environment/troubleshooting/`** | 问题与解决 | CRLF、`bash\r`、Git 合并与未跟踪冲突等 |
| **`docs/research/`** | 研究笔记 | SSGS、§7、检索头叙事 |

## 单一权威（避免双写）

| 主题 | 唯一权威 |
|------|----------|
| **时间序主线：阶段 0→5、L1 门闩、任务 A/B 选择** | [`overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md`](overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md) |
| **L1 JSON `kind`、L2/L3 分层、检查单、CSV 列** | [`overview/planning/E2E_MAIN_RESULTS_AND_EVIDENCE_TIERS.md`](overview/planning/E2E_MAIN_RESULTS_AND_EVIDENCE_TIERS.md) |
| 叙事框架、基线对照、A/B/C/X、月级阶段表、**§1.0 主验证轴/副线** | [`overview/planning/PROJECT_MASTER_PLAN.md`](overview/planning/PROJECT_MASTER_PLAN.md) |
| **主线叙事链**（树→路径→reader→导航；**SSGS**；数据/登记索引；**§0 中/英摘要可用稿**） | [`overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md`](overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md) |
| **现状**大表、七轴、§3.5 证据梯、决策 §4、公平性 §6 | [`overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`](overview/planning/RESEARCH_STATUS_AND_DIRECTION.md) |
| **阶段 0→结题**（每阶段实验 + 成功标准；与 L1–L4 对齐） | [`overview/planning/RESEARCH_PHASES_0_TO_DONE.md`](overview/planning/RESEARCH_PHASES_0_TO_DONE.md) |
| 轨道里程碑、**当前收口清单**、P0–P3、后备推进 § | [`overview/execution/NEXT_RESEARCH_PLAN.md`](overview/execution/NEXT_RESEARCH_PLAN.md) |
| **本周**勾选、阻塞 | [`overview/execution/CURRENT_SPRINT.md`](overview/execution/CURRENT_SPRINT.md) |
| **从现在→结题** 多步方向（成文 / 冻结 / M2 / 截稿） | [`overview/execution/PLAN_NOW_TO_DONE.md`](overview/execution/PLAN_NOW_TO_DONE.md) |
| 本机 5060 **可复制命令** | [`environment/runbooks/LOCAL_5060_RUNBOOK.md`](environment/runbooks/LOCAL_5060_RUNBOOK.md) |
| 云端扫参 shell、§7 depth、命令模板 | [`environment/runbooks/SERVER_SWEEP_RUNBOOK.md`](environment/runbooks/SERVER_SWEEP_RUNBOOK.md) + [`environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md`](environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md)（**§12** 阶段 5 聚合/测试） |
| 实验 id、路径、结论一行 | [`experiments/planning/EXPERIMENT_REGISTRY.md`](experiments/planning/EXPERIMENT_REGISTRY.md) |
| **2026-04 服务器批次** JSON / CSV 索引（M1、SSGS、leavescale、B-S2+） | [`experiments/planning/DATA_ARCHIVE_202604_SERVER.md`](experiments/planning/DATA_ARCHIVE_202604_SERVER.md) |
| 周历**模板**与历史 | [`overview/planning/ROADMAP.md`](overview/planning/ROADMAP.md)（勾选以 **CURRENT_SPRINT** 为准） |

## 0. 三分钟：该读哪份？

| 目的 | 文件 |
|------|------|
| **从现在 → L1 端到端跑完**（阶段勾选、门闩） | [`overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md`](overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md) |
| **接下来几周干什么**（阶段 5→投出→审稿） | [`overview/execution/PLAN_NOW_TO_DONE.md`](overview/execution/PLAN_NOW_TO_DONE.md) |
| **RAPTOR 式下一重点**（× SSGS × Mamba，阶段 R1–R4） | [`overview/planning/RAPTOR_INSPIRED_NEXT_PHASE.md`](overview/planning/RAPTOR_INSPIRED_NEXT_PHASE.md) |
| **P0 状态备忘**（§Ⅸ 实证冻结、禁混表、Sprint2 与树/平面） | [`overview/execution/P0_STATUS_MEMO.md`](overview/execution/P0_STATUS_MEMO.md) |
| **实证快照 + 北星框架（F0–F5）+ 下一步命令** | [`overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md`](overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md) **§9–§11** + [`overview/execution/PLAN_NOW_TO_DONE.md`](overview/execution/PLAN_NOW_TO_DONE.md) **§Ⅶ** |
| **动机与架构叙事参照**（Mamba×树×SSGS、风险、文献索引） | [`overview/planning/MOTIVATION_MAMBA_TREE_RAG_SSGS_REFERENCE.md`](overview/planning/MOTIVATION_MAMBA_TREE_RAG_SSGS_REFERENCE.md) |
| **「论文 1」何时动笔（工程门闩 G1–G5 · 真 TF 可比）** | [`overview/execution/PLAN_NOW_TO_DONE.md`](overview/execution/PLAN_NOW_TO_DONE.md) **§Ⅷ**（与篇首 **战略 B** 同读） |
| **工程北星：执行计划 · Sprint · 文档/脚本/结果边界** | [`overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md`](overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md)（**唯一主文件**；**RW 动机** 见 [`research/RESEARCH_NOTES.md`](research/RESEARCH_NOTES.md) **§8**） |
| **基础阶段结论（论文格式浓缩）** | [`experiments/phases/FOUNDATION_STAGE_FORMAL_SUMMARY.md`](experiments/phases/FOUNDATION_STAGE_FORMAL_SUMMARY.md) |
| **投稿摘要可用稿（中/英）** | [`overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md`](overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md) **§0** |
| **M1 之后**：更大模型、公平对比、三风险实测 | [`overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`](overview/planning/RESEARCH_STATUS_AND_DIRECTION.md) **§6.3–§6.4** + [`overview/execution/PLAN_NOW_TO_DONE.md`](overview/execution/PLAN_NOW_TO_DONE.md) **§Ⅵ** |
| 现状、证据、推荐推进顺序 | [`overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`](overview/planning/RESEARCH_STATUS_AND_DIRECTION.md) |
| 本周在做什么、阻塞项 | [`overview/execution/CURRENT_SPRINT.md`](overview/execution/CURRENT_SPRINT.md) |
| 任务拆条、收口清单 | [`overview/execution/NEXT_RESEARCH_PLAN.md`](overview/execution/NEXT_RESEARCH_PLAN.md) |
| **0→结题** 阶段表与阶段 5 勾选 | [`overview/planning/RESEARCH_PHASES_0_TO_DONE.md`](overview/planning/RESEARCH_PHASES_0_TO_DONE.md) |
| **Mamba+树+SSGS** 整合主线（**M1** 工具；**M2** **§6**；**phase-5 收口** **`NARRATIVE` §8**） | [`experiments/planning/SSGS_MAINLINE_M1.md`](experiments/planning/SSGS_MAINLINE_M1.md) + [`overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md`](overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md) **§8** |
| 投稿草稿块与检查清单 | [`overview/execution/SUBMISSION_PACK.md`](overview/execution/SUBMISSION_PACK.md) |
| 实验登记 | [`experiments/planning/EXPERIMENT_REGISTRY.md`](experiments/planning/EXPERIMENT_REGISTRY.md) |
| 四月服务器数据归档（路径 + STAMP） | [`experiments/planning/DATA_ARCHIVE_202604_SERVER.md`](experiments/planning/DATA_ARCHIVE_202604_SERVER.md) |

## 1. 成文与阶段素材（`experiments/phases/`）

| 文件 | 说明 |
|------|------|
| [`phases/PHASE1_MANUSCRIPT.md`](experiments/phases/PHASE1_MANUSCRIPT.md) | 阶段 1 主成稿（含 §8–§10） |
| [`phases/FOUNDATION_STAGE_FORMAL_SUMMARY.md`](experiments/phases/FOUNDATION_STAGE_FORMAL_SUMMARY.md) | **基础阶段** 正式论文体例 **浓缩总结**（摘要/方法/结果/讨论/结论；七轴分列） |
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

---

## 7. 文档冗余说明（2026-04 整理）

以下文件 **仍有价值**，但与主线 **部分重叠**；**新决策以「单一权威」表为准**，勿双写数字。

| 易重复读 | 实际请优先读 |
|----------|----------------|
| **`NEXT_RESEARCH_PLAN.md`** 全篇当路线图 | **`MASTER_EXPERIMENT_PLAN_E2E.md`**（时间序）+ **`CURRENT_SPRINT.md`**（精简执行面板） |
| **`NEXT_RESEARCH_SPRINT_AFTER_MANUSCRIPT.md`** 当总计划 | **R4 Wikitext 服务器队列** 专用；总骨架 **MASTER §4** |
| **`END_TO_END_TASK_PIPELINE.md`** 当 L1 定义 | **T1/T2/T3 工程细节**；**L1 正式 `kind`** 以 **`E2E_MAIN_RESULTS_AND_EVIDENCE_TIERS.md` §7** 为准 |
| **`RESEARCH_PHASES_0_TO_DONE.md`** 与 **MASTER** | **阶段 0–7 历史与成功标准**；**L1 门闩** 以 **MASTER 阶段 5** 为硬条件 |
| **`PHASE1_MANUSCRIPT.md`** vs **`SUBMISSION_STYLE_MANUSCRIPT_DRAFT.md`** | **IMRaD 投稿体例** 以 **`SUBMISSION_STYLE_MANUSCRIPT_DRAFT.md`** + **`latex/submission_manuscript.tex`**；**PHASE1** 为阶段 1 **存档叙述** |
| **`FOUNDATION_STAGE_FORMAL_SUMMARY.md`** vs **NARRATIVE §0** | **浓缩成稿** vs **摘要可用稿**；改摘要时 **只改一处真源**（建议 **NARRATIVE §0** 或 **SUBMISSION 草案**，另一处 **指向**） |
| **`ROADMAP.md`** 周历 | **模板与历史**；**当周勾选** 以 **`CURRENT_SPRINT.md`** 为准 |

**成稿与路径核对**：**`SUBMISSION_PACK.md` §A2** + **`EXPERIMENT_REGISTRY.md`**（basename **逐字**）。
