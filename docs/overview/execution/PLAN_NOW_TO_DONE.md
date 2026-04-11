# 从现在到结题：多步总览（滚动）

> **用途**：给你一张 **方向卡** — 每步 **做什么、验证什么、产出什么**；细节命令仍以 **`NEXT_EXPERIMENTS_COMMANDS.md`**、**`SSGS_MAINLINE_M1.md` §6**、**`SUBMISSION_PACK.md`** 为准。**实证快照与推荐命令** → **`NARRATIVE_MAINLINE_TREE_READER_SSGS.md` §9–§11**；**可落地框架 F0–F5** → 下文 **§Ⅶ** 与 **`NARRATIVE` §10**。  
> **两条战略并存**：（**A**）**尽快收束成稿**：实证已够时走 **§Ⅰ–§Ⅳ**（阶段 5）。（**B**）**工程北星优先**：把 **「第一篇论文」的正稿启动** 推迟到 **§Ⅷ 门闩**（全流程真可用 + 与 **真实 Transformer 栈** 公平可比）之后；下文 **§Ⅰ 默认「下一步优先」仅适用于战略 A** — 若你选 **B**，以 **§Ⅷ** 为主轴，**勿**把阶段 5 成稿当成当前里程碑。  
> **当前选定（2026-04-11）**：**战略 B** — **工程门闩后再启动论文 1 主稿（§Ⅰ）**。门闩交付见 **`ENGINEERING_NORTH_STAR_PLAN.md`**；**动笔前** 须 **GitHub Actions `engineering-tests` 对推送 HEAD 为绿**（与本地 **`py -3 -m pytest` … engineering 四测** 一致）。  
> **战略 B · 门闩 G 之后（2026-04 修订）**：**下一里程碑不是 §Ⅰ 成文**，而是 **验证 Mamba + 树状 RAG + SSGS 在实际任务上的端到端可用性与效果**（任务级指标、与强基线对照）。见下文 **§Ⅸ**。  
> **当前逻辑阶段（战略 A）**：**阶段 5（成文）**（**`RESEARCH_PHASES_0_TO_DONE.md`**）；实证 **0–4** 已收口，**主线不再依赖新实验也能投稿**。  
> **M1 ①②③ 玩具全链已通** 之后若追求 **可用性 / 更大模型 / 更公平对比 / 三风险实测** — 见 **§Ⅵ** 与 **`RESEARCH_STATUS_AND_DIRECTION.md` §6.4**（**与阶段 5 并行规划，不替代投稿闭环**）。  
> **修订**：截稿或每完成一大步，在文末 **修订记录** 写一行；**`CURRENT_SPRINT.md`** 篇首可互链本文件。

---

## 总流程（五段）

```mermaid
flowchart LR
  A[阶段5 成文] --> B[数据冻结]
  B --> C[可选 M2 实验]
  C --> D[截稿 / 投出]
  D --> E[审稿 / 远期]
```

| 段 | 名称 | 你要验证什么 | 完成标志（客观） |
|----|------|----------------|------------------|
| **Ⅰ** | **成文 P0** | 叙事 **≤ L1–L3**（**`RESEARCH_STATUS` §3.5**）；**七轴不混读** | 正稿含 **§A3 类脚注**；**§A2 basename** 与登记册 **人工逐字** 一致 |
| **Ⅱ** | **数据与仓库** | 可复现路径、**`json_path`** 为仓内 POSIX、单测绿 | **`git status`** 干净；**`aggregate_*` stdout** 的 **N** 与 CSV 一致；**AutoDL** **`python -m pytest tests/ -q`** 通过 |
| **Ⅲ** | **可选 M2**（与 Ⅰ 并行） | **M1** 在 **c12** 或 **同树 bundle** 上仍 **可实现**；**非**新对比法默认 | 新 **JSON + STAMP**；**`EXPERIMENT_REGISTRY`** / **`DATA_ARCHIVE`** 补一行或一句 |
| **Ⅳ** | **截稿** | 无未引用路径、无跨轴混表 | 按 venue 提交；本地留 **tag 或 commit** 对齐 **`git_sha`** |
| **Ⅴ** | **审稿后 / 远期** | 按意见 **1 格复现** 或 **补脚注**；远期 **新 `kind`** | **阶段 6** 登记表；**阶段 7** 另立项（**`PROJECT_MASTER_PLAN`**） |

---

## Ⅵ 北星增量（**M1 玩具全链之后**；与 **Ⅰ 成文** 并行规划）

> **定位**：回应 **「链条要能用、对比要公平、得上大模型、要测回溯丢信息」** — **方向正确**，但须 **分档** 与 **先定义可比性**，见 **`RESEARCH_STATUS` §6.4** 长文分析。本节只保留 **执行序**。

| 顺序 | 目标 | 验证什么 | 备注 |
|------|------|----------|------|
| **Ⅵ-0** | **写清「公平」定义** | **同一任务**、**同一预算维度**（token？峰值？步数？）下比什么 | **已定提案**：**`ENGINEERING_NORTH_STAR_PLAN.md` §4.4**（**Q1–Q5**）；**待导师会议确认**；与 **M1** **跨臂 wall 不对等** 不矛盾 |
| **Ⅵ-1** | **仍用 path-reader / M1 槽位加规模**（**`RESEARCH_STATUS` §6.3 档 1**） | **dim / 层数 / chunk / 序列** 抬高后 **结论形状是否仍成立** | **同机**、**新 STAMP**；**先于** 换 7B |
| **Ⅵ-2** | **冻结「中等」LM 作 reader**（**档 2**） | **表示与峰值** 进入 **更真实量级**；仍 **分项** path encoder vs 全模型 KV | **新登记行**；**禁止**与 **paper_main_v1** 逐格混表 |
| **Ⅵ-3** | **三风险电池**（**档 1–2 内优先**） | **有损**：restore 后 **任务落差** 或 **长序列 L3 劣化**；**回退**：规则 vs 学习 **同树同任务**；**H2D**：快照 **设备迁移** 打点 | 各条 **新 `kind` 或 JSON 扩展**；对齐 **§3.5** 三风险 |
| **Ⅵ-4** | **端到端大 LM + 树导航**（**档 3**） | **系统级** 可用性 | **48G**、**训练/冻结** 协议先定；**第二篇或 major revision** 更现实 |

**与当前仓库的衔接**：**Ⅵ-1** 可接 **M2 · `dim256` M1**、**path-batch XL**；**Ⅵ-3** 可接 **`tf_kv_trajectory_l3_minimal`** 思路扩展、**A2-S3**、**§7 S4 fromcpu** 叙事；**Ⅵ-2/Ⅵ-4** 需 **新开实验设计节**（尚未写死脚本名）。

---

## Ⅶ 可落地完整框架（**F0–F5**；与 **`NARRATIVE` §10** 同表）

> **目标**：从 **「阶段 5 能投稿的证据链」** 推进到 **「真预训 reader + 真任务 + 可部署」**，**分档**完成；**禁止**与 **paper_main / path-batch 主图** 无脚注混表。

| 阶段 | 名称 | 验证 / 产出 | 与仓库 |
|------|------|----------------|--------|
| **F0** | **框架内核冻结** | 建树 + reader 槽 + **SSGS** + **M1** + 聚合 CSV + 登记册 **一致** | 对应 **§8–§9** 快照；**Ⅰ+Ⅱ** 打勾后可 **tag** |
| **F1** | **公平与任务定义** | 一页纸：**token / 步数 / 峰值** 哪个维度可比；脚注模板 | **Ⅵ-0**；导师对齐 |
| **F2** | **规模与形状** | **dim / 叶数 / chunk** 抬高后叙事是否仍成立 | **Ⅵ-1**；**`SSGS_MAINLINE_M1` §6**、**`NEXT_EXPERIMENTS_COMMANDS` §2–§4** |
| **F3** | **预训 Mamba2 作 path reader** | HF 权重加载 → 同树 **path-batch** 或 **M1** 槽位；**新 `kind`、新登记行** | **Ⅵ-2**；须 **代码 PR + 实验设计**（当前默认 **toy trunk**） |
| **F4** | **三风险与鲁棒** | 有损 restore、学习 vs 规则回退、H2D | **Ⅵ-3**；扩展现有 **L3 / §7** 线 |
| **F5** | **端到端产品级** | 大 LM + 树导航 + 训练；吞吐与成本 | **Ⅵ-4**；**第二篇 / major revision** 量纲 |

**默认执行序（战略 A）**：**F0（Ⅰ+Ⅱ）→ 投稿** → 并行 **F1** 与 **F2（按需一格）** → **F3** 单独立项（**不与**阶段 5 截稿互相阻塞）。

**默认执行序（战略 B · 见 §Ⅷ）**：**F1 → F2 → F3 → 「TF 可比 harness」→ F4 风险电池 → 门闩 G → 论文 1**；**不**以 **§Ⅰ 成稿** 为当前阻塞。

**下一步命令**（摘录；全文 **`NARRATIVE` §11**）：**`aggregate_*` 两条**；**`pytest`**；可选 **`RUN_WIKITEXT_SMOKE=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh`**；可选 **`bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**。

---

## Ⅷ 论文 1 启动门闩（**工程级全流程 + 与真实 Transformer 栈可比**）

> **存档意图**：你选择 **把「真正意义上的第一篇论文」放到工程闭环之后**；本节约束 **计划文档**，避免与 **§Ⅰ「先成稿」** 混读。**门闩未达成前**：允许 **技术备忘录 / 内部报告 / 代码里程碑 tag** 记录现状，**不**把 LaTeX 主稿当作「论文 1 终局」。

### Ⅷ-0 「真正 TF 框架」的操作定义（动笔前先钉死）

| 维度 | 须写清的内容 |
|------|----------------|
| **TF 指什么** | 至少：**HF `transformers` 路径上的因果 LM**（**`AutoModelForCausalLM`** 或你指定的 **Decoder-only** checkpoint）；**玩具 `tf_kv_*`** 可与 **「真 TF 臂」分列**，**禁止**无脚注混成「已打败 PyTorch 生态」 |
| **同一任务** | 与 **SSGS / M1** 对齐的 **树 + DFS / path-batch** 协议；**目标函数**（导航成功率？CE？latency？）**一条表** |
| **同一预算** | **token 步数**、**峰值显存**、**墙钟** 至少锁 **两项** 在对比行里同时报；见 **§Ⅵ-0** |
| **可复现** | **固定 seed、Docker 或 `environment.yml`、一键脚本、登记 `git_sha`**；与 **审稿补实验** 同标准 |

### Ⅷ-1 建议门闩 **G**（全满足 → 才宣布「开始写论文 1」）

| 编号 | 门闩 | 客观完成标志 |
|------|------|----------------|
| **G1** | **统一 Runner** | 同一入口脚本（或 CLI）可跑 **Mamba2 路径 reader 臂** 与 **HF TF 臂**（同树、同路径形状、同 logging schema） |
| **G2** | **预训权重路径** | **至少一条** **HF 预训 / 官方权重** 的 Mamba（或你选定 backbone）接入 **G1**，**非**仅随机初始化 toy |
| **G3** | **真 TF 对照臂** | **G1** 中 TF 臂使用 **§Ⅷ-0** 定义；产出 **可与 SSGS 迹分列** 的 **KV / 缓存 / 重算** 数字或 **声明的等价测量** |
| **G4** | **端到端「可用」** | 从 **语料 → 建树 → 导航/检索任务 → 指标 JSON** 可 **一键** 复现（**CI 或 nightly** 至少跑 smoke） |
| **G5** | **公平性文档** | **一页纸** 写清 **§Ⅷ-0** + 已知 **不对等**（若有）+ **脚注** 模板 |

### Ⅷ-2 风险与建议（野心 ↔ 可持续）

- **范围**：**G1–G5** 已是 **多季度工程**；若并行 **博士毕业 / 产品节点**，把 **G4 smoke** 拆 **每月可 demo** 的里程碑，避免「全做完才见光」。  
- **中间产出**：门闩前可发 **技术报告 / arXiv workshop / 系统论文（非核心主张）** —— **与「论文 1 主结论」分开命名**，避免自我透支。  
- **与仓库现状**：当前 **path-batch + SSGS + M1** = **强基线 + 机制证据**；**不足**以支撑 **G2/G3** —— **正好**作为 **§Ⅷ** 的 **起点代码**，而非终点叙事。

### Ⅷ-3 与 **§Ⅵ / §Ⅶ** 的映射

- **§Ⅷ** = **「论文 1 何时动笔」** 的 **硬门闩**；**§Ⅶ F3–F5** = 门闩内的 **技术阶梯**；**§Ⅵ** = 公平与风险 **内容** 的 **理论来源**。  
- **战略 B** 下 **§Ⅰ–§Ⅳ** **暂停为 P0**；仅当 **G 达成后** 再启用 **§Ⅰ** 作为 **论文 1 成稿流水线**。**G 已于 2026-04-11 达成**（见 **`ENGINEERING_NORTH_STAR_PLAN.md` §3** 与下文 **门闩 G 已达成** 段）；**§Ⅰ 动笔** 前仍须 **确认 CI 绿**。

**执行计划（文档/脚本/结果分列 + Sprint）**：**`docs/overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md`**；**统一 CLI**：**`scripts/engineering/run_engineering.py`**；速查：**`scripts/engineering/README.md`**；工程 JSON 默认目录：**`results/metrics_result/engineering/`**。

> **G5 公平性一页纸（§Ⅷ-1 子门闩）**：**`ENGINEERING_NORTH_STAR_PLAN.md` §4 + §4.2** 已含 **§Ⅷ-0** 所要求的 **TF checkpoint**、**任务**、**预算两项**、**不对等** 与 **可复现命令** —— **G5 表已冻结**（**2026-04-11**）。**§Ⅷ-1 G1（统一入口）**：**`scripts/engineering/run_engineering.py`** 以子命令转发 **path-batch / G3-a / G3-b / M1**（见 **`ENGINEERING_NORTH_STAR_PLAN.md` §2**）；各 **`kind` 仍分 JSON**。

> **G3 预训因果 LM（独立实验）**：**§4.3** — **G3-a** 烟测与 **G3-b** **`engineering_causal_lm_compare`**（**标准 AR vs 树路径**）为 **主文可独立成表** 的证据线，**与 path-batch 玩具表分表**；**不是**「仅为附录脚注」。实现与登记见 **`ENGINEERING_NORTH_STAR_PLAN.md`**。

### 门闩 **G** 已达成（**工程北星 · 2026-04-11**）

**客观标志**：**G5** §4 一页纸已冻结；**G1** **`run_engineering.py`** 四子命令；**G2/G3** 预训 **`from_pretrained`** + **`ENG-*`** 归档（含 G3-b / 消融 / M1→`engineering/`）；**G4** **`.github/workflows/engineering_tests.yml`**（**11** 个工程相关 **`pytest`** 与 CI 对齐）。**战略 B** 下 **门闩 G 已达成** 后，**默认下一里程碑** 见 **§Ⅸ**（**真实任务验证**）；**§Ⅰ 成稿** 在 **§Ⅸ 至少有一条可写进摘要的任务级结论** 之后再排期。**CI**：**以 GitHub `engineering-tests` 绿为最终确认**（推送 **`aac8fce`** 及之后）。

---

## Ⅸ 真实任务验证（**战略 B · 门闩 G 后优先**）

> **定位**：**ENG-*** 与 **§Ⅷ** 解决的是 **协议对齐、分项测量、可复现与 CI**；**不**等同于 **下游任务 SOTA 或产品可用性**。**本节** 把 **「下一步」** 从 **§Ⅰ 成文** 显式改为 **端到端任务效果**。

| 要回答的问题 | 建议验证 | 与仓库衔接 |
|--------------|----------|------------|
| **整条链路是否「能用」** | 固定 **语料 → 建树 → 检索/导航 → 可报告指标**，**一键或短脚本** 复现 | **G4** 已覆盖 smoke；任务级需 **新 `kind` + 登记行** |
| **是否优于朴素基线** | 同预算下对比：**扁平 chunk 检索**、**无 SSGS 的回溯**、**仅 KV 路径** 等 | **`benchmark_ssgs_vs_kv_tree_nav_wikitext`**（M1）、**`probe_path_reader_linear`** / **`task_wikitext_path_pair`** 可作 **任务原型**；须 **统一任务定义**（见 **§Ⅵ-0**） |
| **Mamba+树+RAG 的价值是否体现在任务上** | 至少 **1 个** 域内或公开 bench：**问答 / 多跳 / 分类 / 生成质量**（择一深入） | **`PLAN` §Ⅶ F4–F5**、**`RESEARCH_STATUS` §6.4**；新实验 **禁止**与 **path-batch 主表** 无脚注混读 |

**完成标志（客观）**：**`EXPERIMENT_REGISTRY.md`** 新增 **`TASK-*` 或 `ENG-task-*`** 至少 **一行**（id、commit、**任务名**、**主指标**、**对照臂**、结论）；可选 **内部一页纸** 写清 **与 Wikitext 机制实验的脚注关系**。

**刻意不做（本阶段）**：**LaTeX 主稿 §Ⅰ–Ⅳ** 全文推进；允许 **备忘录 / 登记册 / 脚本 README** 记录设计。

### Ⅸ-1 叙事假设与对照设计（**树状 vs 平面 · 回溯分档**）

> 与既有 **机制证据**（path-batch、M1、§7）**分列**：本节只约束 **任务级** 假设与 **基线矩阵**，避免把 **微基准墙钟** 误读为 **下游收益**。

| 维度 | 工作假设（需在 **TASK-*** 中显式写进「目的」） | 建议对照臂 |
|------|-----------------------------------------------|------------|
| **索引形状** | **长文档 / 多跳 / 主题级** 问题：树状索引（层次主题、路径即证据链）相对 **平面 chunk 池** 应有 **可测优势**（导航成功率、证据一致性、或主题 held-out 上的 gap）。 | **平面**：top‑*k* 块检索 + 同 reader/LM；**树状**：同语料 **`build_bottom_up_text_tree`** 或等价建树，**不**与 **玩具 dim128 path-batch 主表** 无脚注混表。 |
| **索引形状** | **短上下文 / 单跳 / 快答**：平面更简单；树状收益可能 **边际化**，甚至可能 **更差**（DFS/快照开销）。 | 同一任务上 **必须** 报 **平面强基线**；树状作为 **条件激活**（仅长文档/多跳子集）时可 **分层报告**。 |
| **回溯（SSGS）** | **多跳、多步、检索噪声大、知识密集**：完整 DFS + 快照/恢复（**M1 / 全 SSGS 迹**）有 **机制与公平分项** 支撑，任务上预期 **纠错与可恢复性** 价值高。 | **全回溯臂**：与 **`benchmark_ssgs_vs_kv_tree_nav_wikitext`** / **`demo_ssgs_mamba_wikitext`** 协议对齐的 **同树** 运行；**脚注** 写明 **墙钟不对等**（已有 **`note`** 模式）。 |
| **回溯（SSGS）** | **简单单轮、单跳**：全 DFS **过重**；采用 **轻量回溯**（例如 **≤1–2 次** `restore` 或 **cap `rollbacks`/`snapshots`**）控制 **延迟与实现复杂度**。 | **轻量臂**：与全回溯 **同任务、同预算维度**（至少锁 **token 步数上界** + **峰值或墙钟** 之二，见 **§Ⅵ-0** / **`ENGINEERING_NORTH_STAR_PLAN.md` §4.4**）。 |

**综合建议（执行口径）**：用 **二维任务族** 组织实验 —— **{平面, 树} × {无回溯, 轻回溯, 全回溯}**，在 **长文档/多跳子集** 上检验 **树 +（轻或全）回溯** 是否 **同时** 优于 **平面 + 无回溯**；在 **短/单跳子集** 上优先展示 **平面 + 轻回溯** 的 **性价比**，**不**强行宣称树状 **全域** 更优。

### Ⅸ-2 开工序（**Sprint 1 → 3**；与代码里程碑）

| Sprint | 时长（建议） | 交付物 | 客观完成标志 |
|--------|----------------|--------|----------------|
| **1** | **3–5 天** | **任务卡冻结**：选定 **1** 个「长文档/多跳」原型 + **1** 个「短/单跳」原型；主指标 **1–2 个**（如 leaf/path **acc**、**EM**、任务自定义 **success**）；基线表含 **平面 vs 树**、**回溯三档** 至少 **占位行** | **`PLAN`** 本节或 **`CURRENT_SPRINT.md`** 勾选；**导师/合作者** 可对 **任务定义** 拍板 |
| **2** | **1–2 周** | **最小可跑矩阵**：在 **同一语料与同一 LM/reader 预算声明** 下，跑出 **≥4** 个可区分臂（至少含 **平面无回溯** + **树+一种回溯**）；结果 **JSON** 含 **`kind`**、**`git_sha`**、**任务名** | **`EXPERIMENT_REGISTRY.md`** 首条 **`TASK-*`**；`results/metrics_result/` 或子目录 **`task/`** 下 **归档 basename** |
| **3** | **并行** | **可复现与 CI**：**CPU smoke** 或 **单测** 覆盖 **任务入口 CLI**（若新增脚本）；**不**要求覆盖 GPU 全网格 | **新测** 加入 **`.github/workflows/`** 子集 **或** 文档 **「一键命令」**；**`engineering-tests` 仍绿** |

**立即开工（本周动作）**：在 **`CURRENT_SPRINT.md`** 勾选 **§Ⅸ Sprint 1**；代码侧 **优先复用** **`task_wikitext_path_pair`**、**`probe_path_reader_linear`**、**`benchmark_ssgs_vs_kv_tree_nav_wikitext`**、**`demo_ssgs_mamba_wikitext`** —— **新文件** 仅在为 **统一任务 JSON schema** 或 **回溯 cap** 必要时引入。

### Ⅸ-3 Sprint 1 语料决议（**Wikitext 难度梯度；暂缓 Hotpot/MuSiQue**）

> 吸收外部评审共识：**Sprint 1 不换语料**，在 **Wikitext-2** 上通过 **树深度 × 任务粒度** 制造难度档，避免 Hotpot 等引入 **QA 格式 / 领域 / 建树逻辑** 三重 confounder。

| 共识点 | 说明 |
|--------|------|
| **暂缓 Hotpot** | 标准多跳 QA 与 **A2-S3**（叶对 + ridge + held-out **acc**）**评估口径不同**；语料与 **按连续流建树** 的 **`build_bottom_up_text_tree`** 也不对齐 —— **不宜**在 Sprint 1 与既有 Wikitext 数字 **交叉验证**。 |
| **难度梯度（同一语料、同一 reader 槽）** | **浅档**：**`num_leaves` 4–8**、`fanout=2`、**`--cohort sibling`** — 对应「短/单跳」叙事。**深档**：**`num_leaves` 32–128**、**`--cohort root_child`** — 块大小 **`fanout**(depth−1)**，叶对标签依赖 **更粗子树划分**（比 sibling 更接近「路径聚合 / 多跳」直觉），**且** 已在 **`path_pair_geometry`** / **`task_wikitext_path_pair.py`** 实现，**无需**先换 Hotpot。 |
| **「子树 held-out」叙事** | **理想**形态（例如 **左半树 vs 右半树** 二元标签）在仓库中 **尚无** 现成 `cohort`；**第一步**可用 **`root_child` + `leaf_heldout`** 达到 **「子树级一致性 + held-out」** 的 **可操作近似**，再迭代 **自定义 cohort**（需新几何/标签生成）。 |
| **四臂 F0–T1** | **F0（平面）**：**`benchmark_wikitext_tree.py` 当前无 `--flat`** —— **须新实现**（或 **Sprint 2** 再补）；**Sprint 1 可不跑 F0/F1**（与外部建议一致）。**T0 vs T1**：见下 **实现对齐**。 |

**实现对齐（重要）**：**`task_wikitext_path_pair`** 产出 **`ridge_*.*.test_acc`**（路径 reader → 分类）。**`demo_ssgs_mamba_wikitext`** / **M1** 产出 **`snapshots_taken` / `rollbacks` / `wall_s`** 等 **导航–机制** 指标，**默认不是** 同一套 **`test_acc`**。因此 **「T1 准确率显著高于 T0」** 只有在下列之一成立时才是 **合法完成标志**：（**a**）为 SSGS/截断臂构造 **与 A2-S3 同一监督头** 的 **`kind`**（新代码或扩展）；或（**b**）Sprint 1 **拆分报告**：**T0** = **A2-S3 深档 acc**（及 **init-seed×5**）；**T1** = **同树** SSGS/M1 **JSON**（效率迹），**脚注** 写清 **任务目标不同列**，**不**合成虚假「acc 胜负」。推荐 **（a）** 作为 Sprint 2 目标，**（b）** 作为 Sprint 1 **诚实下限**。

**Sprint 1 完成标志（修订版）**：**深档**（例 **`num_leaves=32`**、`fanout=2`、**`cohort root_child`**）上 **`task_wikitext_path_pair`** **`leaf_heldout` + `init-seed` 扫描** 产出 **可登记 JSON**；**可选同构** 跑 **`demo_ssgs_mamba_wikitext`** / **M1** **同参同树** 作为 **回溯代价** 旁列。**若** 实现 **（a）** 后再采用 **「T1 acc > T0 acc」** 为硬门闩。

**风险**（与外部表一致）：深档 **全体接近 chance** → 先用 **depth=4（16 叶）** 试水；平面在深档 **意外优于树** → 记为 **条件性结论**，叙事转向 **「树何时有效」**。

### Ⅸ-4 Sprint 1 交付 **（b）**：可复制命令与脚注模板

> **策略**：**不改脚本**；**Table A** = **`task_wikitext_path_pair`** 的 **`ridge_*.*.test_acc`**（多 **`--init-seed`**）；**Table B** = **同 `num_leaves` / `fanout` / chunk 默认** 的 **`demo_ssgs_mamba_wikitext`** 或 **`benchmark_ssgs_vs_kv_tree_nav_wikitext`** JSON（**导航代价迹**）。两表 **不可做数值胜负**，脚注一句即可。

**CLI 勘误**（相对口头 bash）：**无** `--depth` / `--leaves`；深度由 **`--num-leaves`** 与 **`--fanout`** 推导（须满足 **`num_leaves = fanout ** depth_edges`**（整数））。**`task_wikitext_path_pair.py`**、**`demo_ssgs_mamba_wikitext.py`**、**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** 均在 **`scripts/research/`**（**不是** `scripts/benchmarks/`）。

**`root_child` 块大小**：**`block = fanout ** (depth_edges − 1)`**。例：**`fanout=2`**、**`num_leaves=32`** → **`depth_edges=5`** → **`block=16`**（叶 **0–15** 与 **16–31** 各一块）。**冒烟**：**`num_leaves=8`** → **`depth_edges=3`** → **`block=4`**；若 **`leaf_heldout`** 报 **单类**，改 **`H`** 或 **`stratified`**。

---

**Table A — T0（深档分类 acc；5 seeds）**

```bash
# Bash（仓根；GPU 若可用则去掉 --cpu）
for seed in 0 1 2 3 4; do
  py -3 scripts/research/task_wikitext_path_pair.py \
    --num-leaves 32 --fanout 2 --cohort root_child \
    --init-seed "$seed" --cpu \
    --out-json "results/metrics_result/task_wikitext_f2_n32_rootchild_seed${seed}.json"
done

py -3 scripts/research/aggregate_task_wikitext_path_pair_json.py \
  --glob "results/metrics_result/task_wikitext_f2_n32_rootchild_seed*.json"
```

**`leaf_heldout` 与 `root_child`（两块等大）**：当 **`num_leaves = 2 × block`**（例 **n=32**、**block=16**）时，**前缀 train / 后缀 test** 与 **块边界** 的几何 **不可能** 同时让 **train、test 内** 都存在 **跨块叶对**（故必触发 **`single class in train or test`**）。**证明要点**：train 含跨块对 **⇔** **`n−H−1 ≥ block`** **⇔** **`H ≤ n−block`**；test 含跨块对 **⇔** **`n−H ≤ block−1`** **⇔** **`H ≥ n−block+1`**；二者 **⇒** **`H ≤ n−block` 且 `H ≥ n−block+1`**，矛盾。**结论**：**`cohort=root_child`、`n=32`** 时 **勿用 `leaf_heldout`**；更强 held-out 请 **改用 `stratified` + 调 `test_frac` / `split-seed`**，或 **`cohort=sibling`**（块小、**`leaf_heldout`** 易同时两类），或 **改标签几何**（新代码）。

**Table B — T1（同树导航代价；与 Table A 分列）**

```bash
# SSGS-only（玩具 Mamba + DFS）
py -3 scripts/research/demo_ssgs_mamba_wikitext.py --cpu --num-leaves 32 --fanout 2 \
  --out-json results/metrics_result/ssgs_mamba_wikitext_n32_f2_sprint1b.json

# 或 M1 三臂（CUDA 推荐；墙钟与 TF 臂脚注见脚本 docstring）
py -3 scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py --device cuda \
  --num-leaves 32 --fanout 2 \
  --out-json results/metrics_result/ssgs_vs_kv_wikitext_n32_f2_sprint1b.json
```

**脚注模板（粘贴进 Table B 或总表下）**：「**Table A** 报告 **叶对 + ridge** 的 **test_acc**；**Table B** 报告 **同语料建树** 上 **SSGS/M1 导航** 的 **快照/回退/墙钟/峰值**，**任务目标与 Table A 不同**，**不可**将两表数值直接比较。」

**Sprint 2（a）**：统一监督头后再回答 **「回溯是否提升 acc」**。

### Ⅸ-5 Sprint 1c（**下一步实验**；仍无新代码）

> **动机**：**Sprint 1（b）** 深档 **stratified** 上 **ridge test_acc 近饱和**（见 **`TASK-20260407-wikitext-sprint1b-…`**）— **1c** 用 **held-out 叶**（**在 `root_child` 下不可用**，见上节）与 **浅档** 把 **难度梯度** 做实。

| 序号 | 实验 | 目的 |
|------|------|------|
| **1c-A** | **`n=32`、fanout=2、`cohort=sibling`、`pair_split=leaf_heldout`、`heldout_leaves=8`**（**5×`init-seed`**） | **`sibling` 块=2**，前缀/后缀 **可同时** 含 **同/异块** 叶对；**深档 + 叶级 held-out**（与 **（b）** 的 **root_child 语义不同**，**分列**） |
| **1c-B** | **`n=8`、fanout=2、`cohort=sibling`、`stratified`、5 seeds** | **§Ⅸ** 浅档「短/单跳」占位 |

**服务器 GPU（bash；仓根）**

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES=0

# 1c-A：深档 n32 + sibling + leaf_heldout（勿对 n32 使用 root_child+leaf_heldout）
for seed in 0 1 2 3 4; do
  python3 scripts/research/task_wikitext_path_pair.py \
    --num-leaves 32 --fanout 2 --cohort sibling \
    --pair-split leaf_heldout --heldout-leaves 8 \
    --init-seed "$seed" \
    --out-json "results/metrics_result/task_wikitext_f2_n32_sibling_h8_seed${seed}.json"
done
python3 scripts/research/aggregate_task_wikitext_path_pair_json.py \
  --glob "results/metrics_result/task_wikitext_f2_n32_sibling_h8_seed*.json"

# 1c-B：浅档 sibling n=8
for seed in 0 1 2 3 4; do
  python3 scripts/research/task_wikitext_path_pair.py \
    --num-leaves 8 --fanout 2 --cohort sibling \
    --init-seed "$seed" \
    --out-json "results/metrics_result/task_wikitext_f2_n8_sibling_seed${seed}.json"
done
python3 scripts/research/aggregate_task_wikitext_path_pair_json.py \
  --glob "results/metrics_result/task_wikitext_f2_n8_sibling_seed*.json"
```

**若 1c-A 仍报错**：改 **`H`**（如 **4** / **12**）或 **`--split-seed`** + **`stratified`**。**完成**：**`EXPERIMENT_REGISTRY.md`** 新行 **`TASK-*-sprint1c-*`** + 聚合 **mean±std**。

### Ⅸ-6 Sprint 2 建议序（**滚动；与战略 B 一致**）

> **当前实证小结**（**2026-04**）：**（b）** **root_child·stratified** ≈ **饱和**；**1c-A** **sibling·leaf_heldout** 上 **reader 可区分**、**`init_seed` 有方差**；**Table B**（**SSGS/M1**）**代价迹** 独立成列；**（b）与 1c 分列**、**Table A 与 Table B 分列** —— **足以支撑「继续研究」**，**不足**以宣称 **「回溯已证提升下游 acc」**（须 **§Ⅸ Sprint 2（a）** 或 **等价任务设计**）。

| 优先级 | 方向 | 产出 / 判据 |
|--------|------|-------------|
| **P0** | **叙事 + 方法冻结** | 一页内部 memo：**（b）/ 1c-A / 1c-B / Table B** 各 **一行** 主结论 + **禁止混表** 句；对齐导师 **§Ⅵ-0 / `ENGINEERING_NORTH_STAR_PLAN` §4.4** |
| **P1** | **Sprint 2（a）· 统一监督** | 在 **同 held-out 协议** 下，**SSGS 路径表示** 进 **与 ridge 可比的头**（或 **BCE**）；回答 **「回溯是否提升 acc」** — **新 `kind` + `TASK-*`** |
| **P1b** | **init 协议** | **1c-A** 类任务：**固定 `split_seed`**、报告 **多 seed mean±std** 或 **固定 checkpoint**；避免单次 seed 讲故事 |
| **P2** | **平面基线 F0** | **`benchmark_wikitext_tree` 无 `--flat`** — 需 **小 PR**：**同语料块序列** 对照 **树 path-batch**（**脚注分列**） |
| **P2** | **可选：深档 root_child 再扫** | **`stratified` + `test_frac` / `split_seed` 网格** 一格；**不**与 **1c-A** 无脚注合并 |

**战略 B 下**：**§Ⅰ 成稿** 仍 **不**作为阻塞；**优先** **P0 → P1b**（低成本）→ **P1**（开发窗口）→ **P2**。

---

## Ⅰ 阶段 5 成文（**战略 A 默认下一步优先**）

| 顺序 | 动作 | 验证点 | 产出 |
|------|------|--------|------|
| **Ⅰ-0** | **摘要**：从 **`NARRATIVE_MAINLINE_TREE_READER_SSGS.md` §0** 迁入中/英摘要；与 **`SUBMISSION_PACK` §A1b** 对表 | 摘要含 **path-batch 主结论**、**naive/fused**、**一句 SSGS/M1**、**局限**；**无数值格点堆砌**（除非 venue 要求） | LaTeX/Word **Abstract** 可投审 |
| **Ⅰ-1** | 读 **`SUBMISSION_PACK.md` §A1–A2** + **`FIGURE_CAPTIONS_STAGE1.md`** 七轴表 | 能说出 **主验证轴**（树+Mamba+SSGS/M1）与 **副线**（检索/探针） | 心里有「哪些表绝对不能混」 |
| **Ⅰ-2** | 把 **§A3 / §A3b**、**§A2.1** 粘进 LaTeX/Word | 每个 **测量轴** 在正文或附录 **有脚注或一句边界** | 主稿可给导师/合作者通读 |
| **Ⅰ-3** | 全文检索 **`results/`** 引用 | 每个 **basename** ↔ **`EXPERIMENT_REGISTRY`** **一行** | **§A2 核对表** 全打勾 |
| **Ⅰ-4** | 摘要 / 讨论 / 局限 | 不出现 **L4 级 Agent 已证成**；**三风险** 与 **`PHASE1_MANUSCRIPT` §9.2** 一致 | 可投版本叙事 |

---

## Ⅱ 数据冻结与仓库卫生（**可与 Ⅰ 穿插**）

| 顺序 | 动作 | 验证点 |
|------|------|--------|
| **Ⅱ-1** | 仓根 **`aggregate_ssgs_mamba_wikitext_json.py`** + **`aggregate_ssgs_vs_kv_wikitext_json.py`**（见 **`NEXT_EXPERIMENTS_COMMANDS.md` §12**） | **`json_path`** 列为 **`results/metrics_result/...`**；**stdout `N row(s)`** 记下 |
| **Ⅱ-2** | **`python -m pytest tests/ -q`**（**AutoDL / mamba2**） | 全绿；数字回填 **`NEXT_RESEARCH_PLAN` §1**（若变更） |
| **Ⅱ-3** | **`git add` / `commit` / `push`** | **`git status`** 干净；无 **`metrics_result` 手改脏行** |

---

## Ⅲ 可选实验（**M2**；**不阻塞 Ⅰ**）

> **原则**：已有 **M1 n64 + L3（1617Z）**、**path-batch**、**§7**、**SSGS grid** 时，**只做「有叙事收益」的一条** 即可。

| 顺序 | 实验 | 验证什么 | 命令入口 |
|------|------|----------|----------|
| **Ⅲ-1** | **M1 · `chunk_len=12`**（**n8 或 n64** 选一） | **同 harness** 在 **c12** 下 **三臂仍 `ok`**；与 **A2-S2 c12** **脚注分列** | **`SSGS_MAINLINE_M1.md` §6.2 B3** — 直调 **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py --chunk-len 12`** |
| **Ⅲ-2** | **同树 path-batch + SSGS bundle** | **path-batch** 与 **SSGS** **同一建树** 可跑通一行 JSON | **`run_ssgs_mamba_wikitext_cuda.sh`** **`RUN_WIKITEXT_SMOKE=1`**（**`NEXT_EXPERIMENTS_COMMANDS.md` §10**） |
| **Ⅲ-3** | **`git pull` 后 M1 单格** | **`git_sha`** 与当前代码一致 | **`M1_LEAVES="8"`** + 新 **`M1_STAMP`**（**`run_m1_ssgs_vs_kv_wikitext_cuda.sh`**） |

**默认建议**：若时间紧 — **跳过 Ⅲ**，直接 **Ⅰ + Ⅱ**；若导师要 **「c12 对齐」** — 只做 **Ⅲ-1** 一格。

---

## Ⅳ 截稿与投出

- **冻结**：**commit hash**、**主图 PNG 文件名**、**CSV basename** 写入 **方法/附录**。  
- **自检**：**`SUBMISSION_PACK.md` §A8** 过一遍。  
- **不**在截稿前夜改 **`results/metrics_result/*` 文件名**（除非同步改登记册与全文）。

---

## Ⅴ 审稿后（阶段 6）与远期（阶段 7）

| 类型 | 做什么 | 备注 |
|------|--------|------|
| **审稿补实验** | 点名复现、补 **1 个 STAMP**、补表 | 每条意见 → **登记行或正文修改**（**`RESEARCH_PHASES_0_TO_DONE.md` §阶段 6**） |
| **加分项** | **B-S3**、**训练型 L3**、**RAPTOR 式树** | **新 `kind` / 新 harness**；**禁止**与主表无脚注合并 |

---

## 本周「开工」最小序列（可照抄勾选）

> **若采用战略 B**（**§Ⅷ**）：本周主轴改为 **G1 统一 Runner 草图** + **§Ⅷ-0 一页纸**；下列 **Ⅰ/Ⅱ** 勾选 **可冻结**。

1. [x] **Ⅰ-0**：摘要 — **已并入** **`PHASE1_MANUSCRIPT.md`** **摘要 / §7 英文**（**`NARRATIVE` §0**）；终稿 LaTeX 再粘贴；与 **`SUBMISSION_PACK` §A1b** 对表。  
2. [x] **Ⅰ-2**：**§A3** 入稿 — 见 **`PHASE1_MANUSCRIPT.md` §5.2**（与 **`SUBMISSION_PACK` §A3/§A3b** 同步）。  
3. [x] **Ⅱ-1**：本机/仓根 **重聚合** 两个 **grid CSV**（**2026-04-11** 已跑：`ssgs_*_nav_grid`、`ssgs_mamba_wikitext_grid`）。  
4. [x] **Ⅱ-2**：**AutoDL** **`python -m pytest tests/ -q`** — **28 passed**, **4 subtests**（**2026-04-11**）。  
5. [x] **Ⅲ-1**（**M1 `chunk_len=12` n8**）已归档；**Ⅲ-2**（**`RUN_WIKITEXT_SMOKE=1`** 同树 bundle）仍 **可选**（见 **`NARRATIVE` §6**）。  
6. [ ] **Ⅵ-0**：与导师 **确认** **`ENGINEERING_NORTH_STAR_PLAN.md` §4.4**（**Q1–Q5** 已定稿提案；会后若有改动须同步该节）（再排 **Ⅵ-1**）；**讨论口径** 可压缩进方法脚注。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：Ⅰ–Ⅴ 段 + M2 可选序 + 本周最小勾选 |
| 2026-04-11 | **§Ⅵ**：**M1 之后** 北星增量（规模 / 大模型分档 / 三风险 / 公平定义）；互链 **`RESEARCH_STATUS` §6.4**；勾选 **Ⅵ-0** |
| 2026-04-11 | **本周序列**：**Ⅰ-2** / **Ⅱ-1** / **Ⅱ-2** / **Ⅲ-1** 勾选；**§5.2** 入 **`PHASE1_MANUSCRIPT`** |
| 2026-04-11 | **§Ⅰ**：增 **Ⅰ-0**（摘要 ← **`NARRATIVE_…` §0**）；**`NARRATIVE` §6** 主线下一步（P0 / Ⅲ-2 / Ⅵ-1） |
| 2026-04-11 | **§Ⅶ**：**F0–F5** 可落地框架与 **`NARRATIVE` §10** 对齐；**下一步命令** 摘录 |
| 2026-04-11 | **篇首**：**战略 A / B** 并存；**§Ⅷ** — **论文 1 门闩**（**G1–G5**、真 **TF** 定义）；**B** 下 **§Ⅰ 非 P0** |
| 2026-04-11 | **§Ⅷ 后**：互链 **`ENGINEERING_NORTH_STAR_PLAN.md`**（**Sprint**、**ENG-***、**`scripts/engineering/`**） |
| 2026-04-11 | **工程北星落地**：**`master`** **`a01d899`** — **G1** path-batch 信封、**G3** causal KV smoke、**G4** **`.github/workflows/engineering_tests.yml`**；**GitHub** 上确认 **Actions** **`engineering-tests`** 通过；登记 **`EXPERIMENT_REGISTRY`** **`ENG-*`** |
| 2026-04-11 | **§Ⅷ G5**：**`ENGINEERING_NORTH_STAR_PLAN.md` §4 + §4.2** 填满 **§Ⅷ-0** 四维度；**G5 公平性一页纸已冻结**（旁注见 **§Ⅷ** **`ENGINEERING…` 链** 段末） |
| 2026-04-11 | **G3**：**§4.3** — **独立实验**（**G3-a** / **G3-b**）；与 **玩具 path-batch** **分表**；旁注 **§Ⅷ** |
| 2026-04-11 | **§Ⅵ-0**：互链 **`ENGINEERING_NORTH_STAR_PLAN.md` §4.4**（公平 **Q1–Q5** 提案） |
| 2026-04-11 | **§Ⅷ-1 G1**：**`run_engineering.py`** 统一 CLI；**§Ⅷ** 段末 **G5** 旁注更新 |
| 2026-04-11 | **战略 B** 锁定；**门闩 G 已达成**（§Ⅷ 新小节）；**§Ⅰ 动笔前** 以 **GitHub `engineering-tests` 绿** 为准（**`aac8fce`**） |
| 2026-04-07 | **§Ⅸ**：门闩 G 后 **优先真实任务验证**（Mamba+树状 RAG+SSGS **端到端效果**），**暂缓** §Ⅰ 成文为主里程碑 |
| 2026-04-07 | **§Ⅸ-1 / §Ⅸ-2**：**树状 vs 平面** 与 **回溯分档**（轻量 **1–2 次** vs 全 DFS）假设 + **Sprint 1–3** 开工序 |
| 2026-04-07 | **§Ⅸ-3**：Sprint 1 **Wikitext 难度梯度**（**暂缓 Hotpot**）；**`root_child`/`sibling`**；**T0 vs T1 acc** 须 **统一任务** 或 **拆分报告**（**`benchmark_wikitext_tree` 无 `--flat`**） |
| 2026-04-07 | **§Ⅸ-4**：Sprint 1 锁定 **（b）** — **Table A/B** 可复制命令、**`scripts/research/`** 路径、**`root_child` block=16 @ n32**、脚注模板 |
| 2026-04-07 | **§Ⅸ-4/5 勘误**：**`root_child` + `leaf_heldout` @ n=32** **不可能** 两类 train/test；**§Ⅸ-5** **1c-A** 改为 **`sibling` + `leaf_heldout` H=8**；登记 **`TASK-20260407-wikitext-sprint1b-…`** |
| 2026-04-07 | **§Ⅸ-6**：**Sprint 2 建议序**（**P0 叙事冻结**、**P1（a）统一监督**、**P1b init 协议**、**P2 平面 / 扫参**） |
| 2026-04-07 | **P0 备忘**：**`P0_STATUS_MEMO.md`**（**§Ⅸ 快照**、**禁混表**、**Sprint2（a）与树/平面不互阻塞**） |
