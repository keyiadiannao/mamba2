# 投稿体例草案（树状路径编码 · Mamba-2 · SSGS · 对照实验）

> **定位**：在 **`PHASE1_MANUSCRIPT.md`**、**`FOUNDATION_STAGE_FORMAL_SUMMARY.md`** 与 **`NARRATIVE_MAINLINE_TREE_READER_SSGS.md` §0** 的事实基础上，整理为**可直接扩写 LaTeX** 的 **IMRaD 体例**；**数字与 basename** 以 **`EXPERIMENT_REGISTRY.md`**、**`DATA_ARCHIVE_202604_SERVER.md`** 及仓内 **`results/metrics_result/`** 为准。聚合表 **行数 N** 以仓根 **`aggregate_ssgs_*.py` 的 stdout** 为准。

---

## Title（英文）

**Measured Path-Batch Efficiency and Navigation-Style Backtracking on Tree-Structured Text: Mamba-2 Readers, SSGS, and Controlled TF-KV Baselines**

---

## Abstract（English, ~200 words）

Tree-structured retrieval often reduces to encoding text along root-to-leaf paths. We present a **unified measurement program** for this setting: a **path-batch harness** that swaps **Transformer, GRU, and Mamba-2** path readers in the same slot, reporting **per-step latency** and **CUDA peak memory** under fixed build-and-benchmark protocols. On **RTX 3090**, **HuggingFace-style Mamba-2 without fused kernels** reaches **GiB-scale** peaks on the same grids where **Transformer/GRU** remain near **hundreds of MiB**; with **`mamba_ssm` fused kernels**, Mamba-2 peaks fall to **tens–hundreds of MiB**, often **about two orders of magnitude** below naive—demonstrating that **reported efficiency is strongly implementation-dependent** and must be tied to **naive vs fused** and **machine context**. We extend the **same harness** to **Wikitext-2 shallow trees** across **leaf-count and dimension** grids archived in this repository. Complementing wall-clock tables, we report **SSGS** (**State-Snapshot Guided Search**) on **DFS-ordered** navigation with **snapshot/rollback counts** on the **same tree build**, and **Phase M1** on the **same tree and DFS goal**, comparing **SSGS+Mamba** with **toy TF-KV** arms (**full KV clone** vs **truncate_kv**), including optional **L3** probes. A **minimal trajectory L3** study reports **near-identity** hidden-state agreement between controlled wrong-branch restore and gold-path reference on the toy TF-KV arms. Together, the work delivers **replicable JSON/CSV artifacts** and a **multi-axis** view: **path-batch efficiency**, **DFS navigation traces**, and **path-level KV-style baselines**—each with **explicit table/footnote separation**.

**Keywords**: tree-structured retrieval; path-batch encoding; Mamba-2; implementation sensitivity; state snapshots; SSGS; controlled comparison

---

## 摘要（中文，与英文同轴）

树状索引下，检索与导航常归结为沿根—叶路径对文本块做序列编码。本文给出**可复现的成套测量**：在固定**建树与 path-batch 协议**下，于**同一 reader 槽位**接入 **Transformer、GRU、Mamba-2**，系统报告**步均耗时**与 **CUDA 峰值显存**。在 **RTX 3090** 上，**无融合核的 HF 风格 Mamba-2** 可在与对照相同的网格上达到 **GiB 量级**峰值，而 **Transformer/GRU** 多停留在 **百 MiB** 量级；在启用 **`mamba_ssm` 融合核**后，Mamba-2 峰值常降至 **数十至数百 MiB**，相对 naive 往往可达**约两个数量级**的落差——表明**可观测效率高度依赖实现路径**，主文须**同机、naive/fused 分列**呈现。我们在 **Wikitext-2 浅树**上沿用**同一 harness** 扩展**叶数与维度**网格，并登记归档。与墙钟主表并列，本文报告 **SSGS**：在 **DFS 试错序**下基于 **Mamba `DynamicCache`** 的**快照/回滚计数**（与 path-batch **分列测量**）；以及 **Phase M1**：在**同建树、同 DFS 目标叶**上，将 **SSGS+Mamba** 与 **玩具 TF-KV** 两臂（**clone** / **truncate_kv**）并表，含可选 **L3**（隐状态、固定叶头 CE）。另含 **L3 轨迹最小对照**，在受控错枝与金路径之间报告**表示层高度一致**的读数。上述结果以**多份 JSON 与汇总 CSV** 形式固化于仓库，形成**主效率—导航迹—路径级 KV 类对照**的**并列证据链**。

**关键词**：树状检索；路径批量编码；Mamba-2；实现敏感性；状态快照；SSGS；对照实验

---

## 1 引言

树形组织广泛用于层次化检索与长文档导航：候选答案或证据往往对应**从根到叶的 chunk 序列**。在固定索引结构之上，**路径上的序列编码器**决定每一步的前向代价与显存轮廓。近年来 **Mamba/SSM** 类模型以**固定维度的递推状态**提供与「随序列长度增长的 KV 缓存」不同的**工程叙事**；然而，在真实 **GPU** 与 **HuggingFace** 栈上，**融合核是否启用**等因素可使 **Mamba-2 的可观测峰值**相对 **naive 路径**出现**数量级差异**。若仅依据渐近复杂度或模块名称推断「必然省显存」，将与**实测**相矛盾。

**本文工作**围绕一条清晰主线展开：**在同一套可复现协议下**，系统刻画（1）**path-batch** 上三类 reader 的**延迟与峰值**；（2）**Wikitext-2 真语料浅树**上**同一 harness** 的扩展曲线；（3）**DFS 式导航**下 **SSGS** 的**快照/回滚迹**；（4）**同树同 DFS 任务**上 **SSGS 与玩具 TF-KV 三臂**的**路径级代价与可选 L3 对齐读数**；（5）**§7 玩具协议**与 **L3 轨迹最小实验**提供的**机制分列毫秒/字节**。我们**以登记与聚合表支撑每一条结论**，使审稿人与后续工作能够**逐表复核**。

---

## 2 问题设定与贡献

**任务层**：给定**树索引**与一批**根—叶路径**，在 **path-batch** 设定下对路径上的 chunk 嵌入做**批量前向**；另在 **DFS 试错序**下执行**树上导航**，需要**回溯**时比较**隐状态快照**与 **KV 类策略**的**可测代价**。

**本文贡献**可概括为：

1. **主结果（path-batch）**：在 **3090** 上完成 **naive 与 fused 成对**扫描，主图与 CSV 展示 **Mamba-2 峰值对实现路径的极端敏感性**；并给出 **Wikitext** 上**同 harness** 的 **leavescale / dim256** 等扩展（详见 **`EXPERIMENT_REGISTRY`** 中 **A-stage2-***、**A-20260408-paper-main-3090-*** 等）。  
2. **导航与对照（SSGS + M1）**：在 **与 `benchmark_wikitext_tree` 对齐的建树**上运行 **`dfs_ssgs_mamba`**，输出 **`snapshots_taken` / `rollbacks` / `leaf_checks`** 等**结构化迹**；在**同一棵树、同一 DFS 目标**上运行 **M1 三臂**（**SSGS+Mamba**、**TF-KV clone**、**TF-KV truncate**），报告 **wall 与峰值**等，并可选 **L3 隐状态/下游 CE**（汇总 **`ssgs_mamba_wikitext_grid.csv`**、**`ssgs_vs_kv_wikitext_nav_grid.csv`**，行数 **N** 以聚合 **stdout** 为准）。  
3. **机制补充**：**§7** 各列 **S1–S4** 毫秒与字节；**`tf_kv_trajectory_l3_minimal`** 在**错枝+restore**与**金路径**之间给出 **cosine≈1、L2=0** 级的**表示层一致性**读数（登记 **X-20260411-tf-kv-trajectory-l3-minimal**）。  
4. **效果 proxy（辅线）**：**A2-S3** 叶对 cohort + **岭回归** 等，与 **path-batch 墙钟分列**。

---

## 3 方法概要

**树与数据**：合成平衡树与 **Wikitext-2 叶块 → 自底向上建树**（与 **`benchmark_wikitext_tree` / `demo_ssgs_mamba_wikitext`** 协议对齐处见代码与登记）。

**Path-batch**：对路径集合调用 **`run_tree_reader_benchmark`** 类流程，记录 **`per_step_s`、`peak_alloc_mib`**（CUDA：**`max_memory_allocated`** 单次基准内增量峰值）。**3090** 上 **fused** 与 **`mamba2_naive`** **同网格、同 `WARMUP`/`REPS`** 成对运行，产出 **`paper_main_*_paper_main_{v1,naive_v1}.csv`** 与 **naive vs fused** 主图。

**SSGS**：**`dfs_ssgs_mamba`** — **token 步进** + **`DynamicCache`** 的 **clone/zero_/copy_**；**Wikitext 同建树** 结果归档 **`ssgs_mamba_wikitext_*.json`**，汇总 **`aggregate_ssgs_mamba_wikitext_json.py`**。

**Phase M1**：**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`**，`kind=ssgs_vs_kv_tree_nav_wikitext` — **三臂**同上；可选 **`--l3-tf-kv-hidden`**、**`--l3-tf-kv-downstream-ce`**；汇总 **`aggregate_ssgs_vs_kv_wikitext_json.py`**。

**§7 与 L3 轨迹**：分列测量；**L3 minimal** 独立于 **M1 全 DFS**。

---

## 4 主要结果（陈述式）

### 4.1 Path-batch：实现路径决定可观测峰值

在 **3090** 主网格上，**naive Mamba-2** 与 **fused Mamba-2** 的 **CUDA 峰值**可相差**约两个数量级**量级；**Transformer/GRU** 在**同一网格**上稳定在 **百 MiB** 量级。这一结果**直接支持**主命题：**在 path-batch 负载下，必须把「naive/fused」与「同机条件」写进表格与图例**；**SSM 名义复杂度不能替代实测曲线**。

### 4.2 Wikitext：同一 harness 下的效率扩展

在 **Wikitext-2 浅树**上，我们沿用**同一 reader 槽位**完成 **叶数扫描**（如 **A-stage2-wikitext-leavescale** 系列 **`20260410T1240Z`** 四档叶数 **`n∈{8,16,32,64}`**、**c8 dim128**）及 **dim256** 等扩展登记。**大叶数**格点上，**Mamba2 峰值**可高于**同 harness 的小型 Transformer encoder**（具体数值见对应 JSON/CSV），与 **§4.1** 共同说明：**相对优劣随叶数、维度与实现路径变化**，**以表为准**。

### 4.3 SSGS：DFS 导航迹的可重复读数

在 **Wikitext 同建树**上，**SSGS** 给出 **DFS 序**下的 **`snapshots_taken`、`rollbacks`、`leaf_checks`**；**n 叶**结构上 **`leaf_checks=n`**、**`snapshots_taken=n−1`** 等**自检关系**在归档 JSON 中**稳定出现**。**`rollbacks` 随叶数增大而上升**的趋势在网格中**可观察**，为**回溯强度**提供**直接、可聚合**的指标（汇总表 **`ssgs_mamba_wikitext_grid.csv`**）。

### 4.4 Phase M1：同树同任务的三臂代价形状

在 **n8 / n16 / n32 / n64** 等多档叶数与多 **STAMP** 归档中，**M1** 三臂均给出 **`ok`** 与 **各臂 wall_s、峰值、KV  nbytes** 等字段；**L3 下游 CE** 在固定随机叶头上 **clone/truncate** 两臂 **`abs_ce_delta→0`** 的读数（如 **`20260410T1247Z`** 系列）表明：在**该探针定义**下，**走错再恢复**与**金路径直达**在 **CE 指标上可对齐**，为**「表示是否走歪」**提供**可量化**支撑。**三臂之间**呈现的是**不同回溯策略的相对代价结构**，而非单一「胜率」标签；**这正是 M1  harness 的设计价值**。

### 4.5 L3 轨迹最小对照

**`tf_kv_trajectory_l3_minimal`**（例 **`…20260410T1341Z.json`**）报告 **clone** 与 **truncate_kv** 路径上 **hidden 余弦接近 1、L2 差为 0**，在**最小玩具拓扑**上**钉住**「恢复后与参考前向一致」的**表示层事实**。

---

## 5 讨论

**本文的核心价值**在于：**用同一套开放脚本与登记体系**，把 **path-batch 效率**、**DFS 导航迹**、**路径级 KV 类对照** 与 **机制毫秒** **分层固化**为 **JSON/CSV**，便于**复现与扩展**。**Mamba-2** 的主线结论是 **正向且强的**：**实测效率由实现栈与网格共同决定**，**融合核带来数量级量级的峰值改善**——这是对**工程实践**有直接指导意义的表述。

**M1 三臂**的价值在于**在固定 DFS 任务上并列三种可运行策略**，并给出 **wall、峰值、KV 体积、L3** 等**多维读数**；**即使某一臂在 wall 上不占优**，**并表本身**仍回答「**在同一协议下各策略付出何种代价**」——这是**刻画性（characterization）**贡献，与 **path-batch 上谁更快** **并列呈现、分表理解**。

**外推边界**：本文 reader 为**小宽度编码器**；**全规模解码栈 KV 会计**与**端到端 Agent 系统**属于**更大规模论文**的设定。本文在**方法上**已用**分列轴**与**登记 id** 为后续 **HF 级预训骨干、同 API 真 TF 对照** 预留了**接口式**位置（见 **`PLAN_NOW_TO_DONE.md` §Ⅷ** 的工程门闩叙述）。

---

## 6 结论

我们在**树状路径编码**设定下，完成了 **path-batch 主实验**（**naive/fused 成对、Wikitext 扩展**）、**SSGS 导航迹**、**M1 三臂对照**与 **§7 / L3 轨迹** 的**成套归档**。**主要实证结论**是：**Mamba-2 的可观测 GPU 峰值对融合实现极为敏感**；在此之上，**SSGS 与 M1** 提供了 **DFS 回溯**与 **KV 类策略** 的**可重复、可分栏**证据。该套结果**适合作为正式投稿长文或短文的主体素材**；**图表与 basename** 以 **`EXPERIMENT_REGISTRY`** 与 **`SUBMISSION_PACK.md` §A2** 核对为准。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：投稿体例 **IMRaD** 草案；正面陈述贡献与结果；与 **`PHASE1` / `FOUNDATION` / `NARRATIVE` §0** 同轴 |
