# 基础阶段研究报告（正式体例浓缩稿）

**定位**：将仓库 **阶段 0–4**（基建 → 系统 path-batch → 真语料浅树 → 快照/回溯叙事与 M1 → 检索头探针底线）的 **可审稿结论** 整理为 **IMRaD 式** 短文，便于对外汇报或扩写为期刊/会议长文。**本文是浓缩与体裁示范，不替代** 完整素材 **`PHASE1_MANUSCRIPT.md`**（含 §8–§10 细节）、**`SUBMISSION_PACK.md`**（脚注草稿）与 **`FIGURE_CAPTIONS_STAGE1.md`**（图注与七轴表）。

**证据与路径真相源**：**`docs/experiments/planning/EXPERIMENT_REGISTRY.md`**；**2026-04 服务器批次索引**：**`docs/experiments/planning/DATA_ARCHIVE_202604_SERVER.md`**。**主验证轴 / 副线**：**`PROJECT_MASTER_PLAN.md` §1.0**。**概念链（树→路径→reader→导航）与数据索引表**：**`docs/overview/planning/NARRATIVE_MAINLINE_TREE_READER_SSGS.md`**（**§0** 中/英摘要与 **`PHASE1_MANUSCRIPT`** 摘要/§7 **同步**；**§7** 含 **AutoDL Ⅲ-2** 命令）。

---

## 摘要

树状索引下，以 **固定建树与 path-batch 协议** 对 **Transformer、GRU、Mamba-2** 三类 **路径 reader** 做批量前向，在 **RTX 3090** 上 **同网格、同提交** 对照 **HuggingFace 无融合核（naive）** 与 **`mamba_ssm` 融合（fused）**。**主要发现**：Mamba-2 的 **CUDA 峰值显存** 对实现路径 **极为敏感**——naive 可达 **GiB** 量级，与 **Transformer/GRU** 的 **百 MiB** 量级形成反差；fused 下同网格峰值可降至 **约数十至数百 MiB**（随叶数与维度仍可能上升），降幅可达 **约两个数量级**。故在 **path-batch** 负载下 **不能** 以 SSM 名义复杂度替代 **实测** 效率；主文须 **显式标注 naive/fused 与同机条件**。在 **Wikitext-2 浅树** 上沿用 **同一 reader 槽位** 扩展 **效率网格**（含 **dim256**、叶数至 **256** 等登记级曲线），并以 **叶对 cohort + 岭回归** 给出与 **墙钟分列** 的 **效果 proxy（A2-S3）**。**独立玩具协议（§7）** 提供 **Mamba cache / TF-KV / restore** 等 **分列毫秒与字节**；**SSGS×Mamba** 在 **DFS 试错序** 下给出 **快照/回滚计数**；**Phase M1** 在 **同 Wikitext 建树、同 DFS 任务** 上并表 **SSGS** 与 **玩具 TF-KV**（full KV clone 与 **truncate_kv**），可选 **L3**（隐状态余弦、固定叶头 CE）；**L3 轨迹最小对照（`tf_kv_trajectory_l3_minimal`）** 在 **硬编码错枝+restore** 与 **金路径直达** 间报告 **表示层余弦**。**检索头** 方向以 **B-S2/B-S2+ 表征探针** 为 **附录级** 证据。**局限**：reader 为 **小宽度编码器**，结论 **不** 自动推广至 **全规模 LLM KV** 与 **端到端 Agent**；**有损状态、回退触发、CPU↔GPU 搬运** 等风险见讨论节。**后续升级路径**：**`RESEARCH_STATUS_AND_DIRECTION.md` §6.3–§6.4**、**`PLAN_NOW_TO_DONE.md` §Ⅵ**。

**关键词**：树状 RAG；路径批量编码；Mamba-2；实现敏感性；状态快照；SSGS；对照实验

---

## Abstract

We study **path-batch encoding** on tree-structured indices with **Transformer, GRU, and Mamba-2 path readers** under a fixed build-and-benchmark protocol. On **RTX 3090**, **naive HuggingFace Mamba-2** can exhibit **GiB-scale CUDA peaks** while **Transformer/GRU** remain near **hundreds of MiB** on the same grids; with **`mamba_ssm` fused kernels**, Mamba-2 peaks drop to **tens–hundreds of MiB** (still **configuration-dependent**), often **~two orders of magnitude** below naive—so **measured efficiency is implementation-dependent** and must not be inferred from nominal SSM complexity alone. **Wikitext-2 shallow trees** extend the **same harness** (including **dim256** and **leaves up to 256** in registry); **A2-S3** adds a **ridge-on-pooled-features leaf-pair proxy** **orthogonal** to wall-clock tables. **§7** reports **per-column toy timings**; **SSGS×Mamba** reports **DFS snapshot/rollback counts**; **Phase M1** tabulates **same-tree DFS** costs for **SSGS vs toy TF-KV** with optional **L3** probes; **trajectory L3 minimal** compares **wrong-branch+restore vs gold** at the **representation** level. **Retrieval-head** work stays at **B-S2/B-S2+ probe** level. **Limitations**: **small encoders**; conclusions **do not** automatically generalize to **full LLM KV** or **L4 Agent claims**; see **§Discussion**.

**Keywords**: tree RAG; path-batch; Mamba-2; implementation sensitivity; state snapshots; SSGS; controlled comparison

---

## 1 引言

**问题**：在 **树形检索/导航** 流水线中，路径上的 **序列编码器** 选择影响 **延迟与显存**。Mamba/SSM 类模型以 **固定大小隐状态** 递推，与 Transformer 系 **随上下文增长的 KV** 在 **代价结构** 上存在 **叙事差异**；但在 **工程实现** 上，**是否启用融合核** 等因子可使 **可观测峰值** 发生 **数量级变化**。

**贡献（基础阶段可陈述范围）**：

1. **系统**：在 **path-batch harness** 下给出 **3090 同机 naive vs fused** 主曲线及 **Wikitext 同 harness** 扩展效率素材（登记见 **EXPERIMENT_REGISTRY**）。  
2. **机制分解**：**§7 玩具协议**（S1–S4 等）将 **clone / restore / TF-R1 / TF-KV** **分列** 测量。  
3. **导航与对照**：**SSGS×Mamba**（**Wikitext 同建树**）与 **Phase M1**（**同树 DFS** 上 **SSGS vs 玩具 TF-KV**）把 **快照式回溯 vs KV 类基线** 做成 **可复现 JSON + 汇总表**；**L3 轨迹最小对照** 补充 **受控错枝/恢复** 的 **表示层** 读数。  
4. **效果 proxy（辅）**：**A2-S3** 在 **同语料浅树** 上给出 **与墙钟分列** 的 **岭回归准确率** 类指标。  
5. **副线**：**B-S2/B-S2+** 讨论 **「检索头」文献** 与 **path reader 表征探针** 的 **边界**（**`RETRIEVAL_HEAD_NOTES.md` §8**）。

**不声称的范围**：**全模型端到端 RAG 吞吐**、**大 LM 公平对打**、**无损 Agent 记忆** —— 见 **§5** 与 **`RESEARCH_STATUS` §3.5**（**L1–L4**）。

---

## 2 相关工作（占位与指针）

树形索引与层次检索（如 RAPTOR/TreeRAG 等）常与 **Transformer LLM** 联用；本文 **聚焦** **树内 path reader 替换** 与 **快照式导航协议**，**不**以本文实验 **覆盖** 平面稠密检索的全部文献。**检索头** 与 **Mamba 路径编码器内部头划分** **非同构**，见 **`RETRIEVAL_HEAD_NOTES.md`**。完整相关工作扩写可在 **`PHASE1_MANUSCRIPT`** 或终稿 **Related Work** 展开。

---

## 3 方法

### 3.1 树与数据

**合成树**：完全 \(k\) 叉、深度 \(d\)，节点为 **`chunk_len × dim`** 嵌入。  
**文本形树**：**Wikitext-2** 叶块 → **自底向上建树**（与 **`benchmark_wikitext_tree`** / **`demo_ssgs_mamba_wikitext`** 同协议处 **显式对齐**）。

### 3.2 Path-batch 主 harness

对给定 **根—叶路径集合** **批量前向**，记录 **`per_step_s`**、**`peak_alloc_mib`**（CUDA：**`torch.cuda.max_memory_allocated`** 在单次基准内的增量峰值）。**3090** 上 **fused** 与 **`mamba2_naive`** 环境 **成对** 运行；登记 **A-20260408-paper-main-3090-{fused,naive,pair}**。

### 3.3 §7 玩具协议

在 **单条或受控** 路径上 **分列** 测量 **Mamba `DynamicCache` clone**、**TF-R1**、**TF-KV 增量**、**restore** 等；各列 **不可互换**（**`RESEARCH_NOTES` §7.0**）。**depth 5–6** 扩展见登记 **X-section7-depth-extension-v1**。

### 3.4 SSGS 与 M1

**SSGS**：**`dfs_ssgs_mamba`** — **token 步进** + **`DynamicCache`** 快照/回滚；**Wikitext 同树** 归档 **`ssgs_mamba_wikitext_grid.csv`**（**`aggregate_ssgs_mamba_wikitext_json.py`**）。  
**M1**：**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`**，**`kind=ssgs_vs_kv_tree_nav_wikitext`** — **同建树、同 DFS 目标叶**，三臂：**SSGS+Mamba**、**TF-KV full clone**、**TF-KV truncate_kv**；可选 **`--l3-tf-kv-hidden`** / **`--l3-tf-kv-downstream-ce`**。汇总 **`ssgs_vs_kv_wikitext_nav_grid.csv`**（**数据行数 N** = 通配 JSON 数，**以聚合 stdout 为准**）。  
**L3 轨迹**：**`tf_kv_trajectory_l3_minimal`** — **≠ M1 全 DFS**；登记 **X-20260411-tf-kv-trajectory-l3-minimal**。

### 3.5 A2-S3 与探针

**A2-S3**：**`task_wikitext_path_pair.py`** — 叶对 cohort，**ridge** 等；与 path-batch **分列**。  
**B-S2+**：**`probe_path_reader_linear.py`** — **合成 topic 叶** 上的 **线性可读性**；**非** 头级因果结论。

---

## 4 结果（按测量轴分列陈述）

以下 **每一条** 对应 **独立测量轴**（**`FIGURE_CAPTIONS_STAGE1.md` 七轴表**）；**禁止**无脚注合并纵轴或互减「一步」。

**R1 — Path-batch 主结果**：**naive vs fused** 峰值可差 **约两个数量级**；**同机 pair** 为 **主文主图**（**`results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`** + **`paper_main_*_paper_main_{v1,naive_v1}.csv`**）。

**R2 — Wikitext 效率扩展**：**阶段 2** 四格/叶扫/**dim256**/**XL 128–256 叶** 等已登记；**5060 naive** 四格为 **动机表**，**禁止**与 **3090 fused** **无标注同表**。

**R3 — §7 分解尺**：S1–S4（含 **branch-truncate demo**）及 **depth 5–6** 归档；**各列毫秒/字节** **仅**在协议内解释。

**R4 — SSGS 计数**：**`snapshots_taken` / `rollbacks` / `leaf_checks`** 随叶数 **上升**；**非** path-batch wall-clock。

**R5 — M1 同树 DFS**：三臂 **wall_s / peak / KV 字节或截断次数** 并表；**跨臂墙钟不对等** 须在正文脚注；**L3** 在玩具 TF-KV 臂上 **CE 对齐（`abs_ce_delta`≈0）** 等已归档（多 **STAMP**）。

**R6 — L3 轨迹**：**`tf_kv_trajectory_l3_minimal`** 报告 **末 hidden 余弦 ≈1**（归档 **CUDA JSON**）。

**R7 — A2-S3**：**init×5** 等与 **TSV 摘要** 已入 **`metrics_result/`**；**test 集小样本** 波动须在正文说明。

**R8 — 探针（辅）**：**5060 CPU** 与 **3090 CUDA** **B-S2+** 归档 **分列**。

---

## 5 讨论

### 5.1 公平性与可比性

在 **「同 path-batch 槽位、小 encoder、同计时定义」** 内，**三 reader** 与 **naive/fused** 对照 **设计合理**（**`RESEARCH_STATUS` §6.1**）。**M1** 中 **Mamba 臂** 与 **玩具 TF-KV 臂** **非同一模型与步长**，**并列** 的是 **代价结构线索**，**不是** **同一 FLOPs 下的赛马**。推广至 **大 LM** 须 **重定义预算与任务**（**§6.3–§6.4**）。

### 5.2 三条硬风险（须在局限中承认）

1. **有损压缩**：隐状态为 **固定维压缩**；**快照 ≠ 无损存档**。  
2. **回退触发**：主线多为 **规则 DFS**；**可学习策略** **未**作为主文承诺。  
3. **H2D / 搬运**：频繁 **CPU↔GPU** 可能抵消 **渐近优势**；§7 **fromcpu restore** 等仅为 **线索**。

### 5.3 证据层级与后续工作

当前素材支撑 **L1–L3 的子集**（**`RESEARCH_STATUS` §3.5**）；**L4 级 Agent 叙事** 须 **新 harness**。**后续**：**`PLAN_NOW_TO_DONE.md` §Ⅵ**（规模档、冻结 LM、风险电池、端到端）。

---

## 6 结论

在 **树路径 path-batch** 设定下，**Mamba-2 reader 的可观测峰值与步时对融合实现极度敏感**；**同机 naive/fused** 是 **主结论的必要条件**。在 **Wikitext 浅树** 上 **同一 harness** 的效率扩展与 **A2-S3 proxy**、以及 **§7 / SSGS / M1 / L3 轨迹** 等多轴素材，共同构成 **基础阶段的系统+机制闭环**，但 **各轴须在正文中脚注分列**。**下一阶段** 若追求 **更大模型与更严格公平对比**，应在 **固定任务与预算** 下 **分档推进**（见 **`RESEARCH_STATUS` §6.4**），**不**与本文主表 **无标注合并**。

---

## 数据与代码

**数据**：**`results/metrics_result/`** 为主归档；部分历史/本机辅数据在 **`results/metrics/`**。**登记**：**`EXPERIMENT_REGISTRY.md`**。**复现命令**：**`NEXT_EXPERIMENTS_COMMANDS.md`**、**`SERVER_SWEEP_RUNBOOK.md`**、**`SSGS_MAINLINE_M1.md`**。

---

## 参考文献（占位）

终稿请按 venue 格式引用 **Retrieval Head**、**Mamba/Mamba-2**、**RAPTOR/TreeRAG** 等；本文档 **不**维护 BibTeX。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：基础阶段 **IMRaD 浓缩稿**；七轴分列、§6.4 指针 |
