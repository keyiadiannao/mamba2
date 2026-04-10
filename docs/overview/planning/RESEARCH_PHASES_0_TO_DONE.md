# 研究全局：阶段 0 → 结题（实验与成功标准）

> **用途**：把「从基建到可投/可答辩闭合」拆成 **可勾选阶段**；每阶段有 **关键实验** 与 **客观成功标准**。  
> **批判性**：外部讨论中的 **动机修辞**（Agent、投资比喻等）**不**计入成功标准；**证据层级** 以 **`RESEARCH_STATUS_AND_DIRECTION.md` §3.5（L1–L4）** 为准。  
> **北星叙事**（快照回溯 × 树状 RAG）见 **`RESEARCH_STATUS_AND_DIRECTION.md` §1.5**；**不等于** 各阶段已自动证成。  
> **滚动维护**：**当前阶段编号** 在 **`docs/overview/execution/CURRENT_SPRINT.md`** 篇首一句；本文件 **少改表结构**，多改 **状态列** 与 **修订记录**。

---

## 阶段总表（与仓库进度对齐 · 2026-04-11）

| 阶段 | 名称 | 关键实验 / 产出 | 成功标准（客观） | 仓库状态 |
|------|------|-------------------|------------------|----------|
| **0** | 基建 | smoke、Git、`EXPERIMENT_REGISTRY` 一行一命令、`git_sha` 进 JSON | 新人可按登记复现一条主链 | **完成** |
| **1** | 系统 path-batch | 3090 同机 naive vs fused；三 reader 网格；主图 CSV/PNG | 主文能写 **实现路径敏感**；5060 与 3090 **分列** | **完成** |
| **2** | 真语料浅树 + 分解尺 + 任务 proxy | Wikitext stage2 网格/叶扫/XL；§7 S1–S4；A2-S3（含 init×5 等） | 正文可并列 **效率** 与 **非墙钟任务指标**，**脚注分列**；局限写明浅树/小 encoder | **完成**（成文时 **子集** 进主文） |
| **3** | 快照/回溯叙事（SSGS + M1 + 玩具 L3） | `dfs_ssgs_mamba`（demo + Wikitext grid）；M1 三臂 + L3 列；`tf_kv_trajectory_l3_minimal` | **L1**：clone/restore 可测；**L2**：M1 同树 DFS **wall/peak/KV** 可表；**L3（部分）**：玩具 TF-KV 轨迹/隐状态 **已登记指标**；**不**外推全模型 Agent | **完成**（到「机制+代价结构」） |
| **4** | 检索头机制 B | B-S2、B-S2+；可选 **B-S3** | 附录或一节 **表征探针**，表述 **非因果头级**；与 **`RETRIEVAL_HEAD_NOTES.md` §8** 一致 | **底线完成**；B-S3 = **加分** |
| **5** | 成文与投稿包 | 无新实验亦可；**SUBMISSION_PACK** §A1–A4 + §A2 路径核对 + §A3 七轴脚注 | **`metrics_result` basename** 与 **`EXPERIMENT_REGISTRY`** 一致；正文结论 **≤ §3.5 的 L1–L3**；L4 仅 **局限/Future Work** | **进行中（主瓶颈）** |
| **6** | 审稿后 / 修订 | 点名复现、S5 表、XL、训练型 L3 新 `kind` 等 | 每条意见 → **登记行或正文修改**；**不**混轴 | **视审稿** |
| **7** | 远期（另立项） | 可学习回退、RL、大 LM、系统 Agent | **新 harness** + **明确任务** 上超越当前 L3 proxy；讨论有损状态/触发器/H2D | **未启动** |

---

## 与 `PROJECT_MASTER_PLAN.md` 周次表的关系

**`PROJECT_MASTER_PLAN.md` §4** 的 **0–6 周次** 是 **月历模板**；本文件的 **阶段 0–7** 是 **结题逻辑序**。大致对应：

- MASTER **0–1** → 本 **0–1**  
- MASTER **2–3** → 本 **2**（+ 部分 **3** 的工程侧）  
- MASTER **4（检索头 B）** → 本 **4**  
- MASTER **5–6** → 本 **5–6**（成文、扩展、回溯叙事定稿）  
- 本 **7** 在 MASTER 中 **未单独成行** → 视为 **Future Work / 第二篇**

---

## 当前应执行（阶段 5 检查清单）

与 **`docs/overview/execution/SUBMISSION_PACK.md`**、**`docs/experiments/planning/DATA_ARCHIVE_202604_SERVER.md` §0** 同步勾选：

1. [ ] **§A2**：主文/附录出现的每个 **`results/metrics_result/`** basename 与 **`EXPERIMENT_REGISTRY`** 一行对齐。  
2. [ ] **§A3 / `FIGURE_CAPTIONS_STAGE1.md`**：**七轴** 各轴一句脚注进正文（含 **M1**、**L3 轨迹玩具** 分列）。  
3. [ ] **`PHASE1_MANUSCRIPT.md`**：摘要/讨论/局限 与 **§3.5** 一致（**不**把动机写成已证结论）。  
4. [ ] **`git status`** 干净；提交前 **`pytest tests/`**（须 torch 的环境）或至少 **`test_aggregate_ssgs_mamba_wikitext_json.py`**。  
5. [ ] （可选）**§7.5 S5**、主图 PNG 入仓策略 — **视截稿**。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：阶段 0–7 表 + 阶段 5 清单 + 与 MASTER_PLAN 映射 |
