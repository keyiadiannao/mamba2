# 项目总体规划（Mamba × 树状 RAG × 检索头）

> **周期**：约 6 个月（2026-04 — 2026-09，可按答辩/截稿调整）  
> **算力**：本地 RTX 5060 8GB（调试与小规模）；AutoDL 48GB（训练与主实验）  
> **代码真相源**：GitHub 仓库；大数据与 checkpoint 不入库，见 `docs/environment/runbooks/SYNC_AND_ENVIRONMENTS.md`

---

## 1. 定位与核心叙事

**一句话**：在**树状结构化检索**（多粒度、多分支导航）场景下，用 **Mamba-2 类状态空间模型** 作为 **reader / 导航器**，结合**检索头**相关机制，在 **同一树索引与同一检索预算** 下，与 **Transformer 系 reader**（及可选的 **混合架构**）对照，度量 **效率（延迟、显存、回溯成本）** 与 **效果（导航/问答）**；并探索 **状态快照式回溯** 等系统级优势。

### 1.0 主验证轴与副线（定调 · 2026-04）

**主验证轴（投稿时优先写进摘要/方法/主结果；证据层级仍受 `RESEARCH_STATUS` §3.5 约束）**

| 支柱 | 本仓落点 | 测量轴（须脚注分列） |
|------|----------|----------------------|
| **树状 RAG** | 合成树 + **Wikitext-2 浅树**；**`benchmark_wikitext_tree` / `run_tree_reader_benchmark`** 同槽位 | **Path-batch 主图**；**阶段 2** 效率网格 |
| **Mamba** | **Mamba2PathReader** vs **Transformer/GRU**；**naive vs fused** 同机对照 | 主文 **Fig.1**、**`paper_main_*` CSV** |
| **SSGS** | **`dfs_ssgs_mamba`** + **`demo_ssgs_mamba_wikitext`**；**`ssgs_mamba_wikitext_grid.csv`** | **SSGS 轴**（快照/回滚计数，**非** path-batch 墙钟） |
| **同树 DFS 对照（M1）** | **`benchmark_ssgs_vs_kv_tree_nav_wikitext`**；**`ssgs_vs_kv_wikitext_nav_grid.csv`** | **M1 轴**（SSGS vs **玩具 TF-KV** clone/truncate；可选 L3） |

**副线（机制与文献接口；不替代主表第一叙事）**

- **检索头 / 探针**：**`RETRIEVAL_HEAD_NOTES.md`**、**B-S2 / B-S2+**（**`probe_retrieval_correlation` / `probe_path_reader_linear`**）—— **附录或独立小节**；回答「与 Retrieval Head 文献的关系」与 **表征线性可读性**，**非** path-batch 纵轴。  
- **B-S3 / 注入训练（C）**：**仅**在审稿点名或单独排期时加投；默认 **不阻塞** 以 **Mamba + 树 + SSGS/M1** 为主线的成文。

**长期北星（超越「阶段 1–2 成文」）**：把 **固定大小隐状态上的 clone/restore** 与 **KV 随上下文线性增长** 在 **树导航、走错路、回溯** 场景下的 **代价结构差异** 讲透；**SSGS（State-Snapshot Guided Search）** 为可包装的算法叙事。**动机层** 可连接 **Agent 式「悔棋」、上下文不被错枝永久污染** 等命题，但 **审稿证据** 必须按 **`RESEARCH_STATUS_AND_DIRECTION.md` §1.5 + §3.5** 的 **L1–L4** 分层推进，**禁止**从效率主图直接跳写到 **无损 Agent 记忆**。

### 1.1 基线与对照组：「Transformer + 平面 RAG」是否必须？

**结论：不是全文必须锚定的唯一对比轴；主故事应优先「树内同设定」的骨干对照。平面 RAG 是强烈推荐的消融/动机基线，而非唯一主基线。**

| 层级 | 作用 | 是否建议作为投稿「主表」 | 说明 |
|------|------|---------------------------|------|
| **A. 主对比（树内、同外圈）** | 同一建树/遍历协议、同一预算下，**Mamba reader vs Transformer reader**（规模与训练条件尽量对齐） | **建议必选至少一条** | 直接对应「状态式路径编码 vs 全局自注意力路径编码」的贡献叙述；与调研报告中 **RAPTOR/TreeRAG 多用 Transformer LLM** 的现状形成**替换式**对照。 |
| **B. 平面 RAG（Top-k 块 / 单向量索引）** | 回答「**为什么要树**」或「树+某 reader 相对传统流水线」 | **建议作消融或辅表** | 对应报告中的 **Flat Retrieval**；用于动机与完整性，但不必与 A 抢主叙事。实现上可用 **同规模 reader + 同语料切块**，仅索引结构不同。 |
| **C. 纯 Mamba 消融** | 变体、深度、状态快照开/关、层数等 | 标配 | 不涉及「和谁说理」，但属于**内部机制**证据。 |
| **D. 混合架构（可选）** | 回应 **HAX / CDSA**、**Retrievit** 等：多查询联合召回、精确定位场景下纯 SSM 的理论与实证边界 | **视结果与篇幅** | 报告引用 **Retrievit**：纯 SSM 在**信息密集、精确定位**任务上常弱于 Transformer，**混合**可更优；若实验显示树导航需要「多分支并行拉取」，可作为讨论或 future work，而非一开始就绑死实现。 |

**「只比较 Mamba、完全不出现 Transformer」可否？**

- **工程/个人验证阶段**：可以，用于快速迭代 Mamba 侧实现。  
- **对外学术叙事**：通常**不够**——审稿人需要知道相对**主流序列骨干**（Transformer 或公开强基线）在**同任务、同预算**下的位置。最低限度：**主表或主图**中应出现与 **Transformer 同设定**的对照，或用 **Retrievit 类文献结论**明确界定「本文聚焦树内替换 reader，与已有「平面检索上 SSM vs Transformer」结论的关系」。  
- **不必**把 **「Transformer + 平面 RAG」** 写成唯一对手；更干净的主线是：**树状 RAG 流水线中 Reader 从 Transformer 换为 Mamba（+检索头增强）**，平面 RAG 作为 **+1 列**说明索引形态的价值。

**与外部调研报告（Mamba + Tree RAG + 检索头）的对齐（要点）**：现有工作已覆盖 **Mamba Retriever + 平面密集检索**；**树状索引 + Mamba 流式遍历**仍属空白主战场——故论文主实验应**落在树上**；**检索头**路线与 **Hidden Attention / Retrieval-Aware Distillation** 等呼应，适合作为机制子线而非替代「树内主对比」。

**论文级贡献候选（从中选 1–2 条做主故事，其余作附录或 future work）**：

1. **系统**：树路径 **path-reader** 设定下，**Mamba-2 vs Transformer vs GRU** 的延迟/显存随 **叶数 batch、chunk、实现（HF naive vs `mamba_ssm` fused）** 的变化；结论已显示 **高度实现敏感**，而非「SSM 必然更省」——详见 **`docs/experiments/planning/PHASE1_VALIDATION_PLAN.md` §6**。  
2. **机制**：**检索头**的发现、注入与与树导航决策的对齐（分析 + 轻量训练）。  
3. **协议**：树导航中的**状态快照 / 廉价回溯**相对 KV 重算或全量重编码的优势边界（**path reader 与全 LLM KV 须分项声明**）。  

   技术展开与 SSGS 算法草图见 **`docs/research/RESEARCH_NOTES.md`**（隐状态 vs KV、实验需固定的对照条件）。

---

## 2. 范围

| 纳入 | 暂不纳入（除非假设被证伪后收窄） |
|------|----------------------------------|
| 玩具树 + 文本形树 +（后续）浅层 RAPTOR 式树 | 超大规模工业索引、多租户服务化 |
| Mamba-2 / SSM reader 与 **同设定下** Transformer（及可选混合）对照；平面 RAG 作消融 | 全量预训练 from scratch |
| 检索头：探针、注入、消融 | 多模态检索 |
| 5060 + AutoDL 可复现实验与 CSV/配置 | 无记录的一次性手跑 |

---

## 3. 工作分解结构（四条线）

| 代号 | 名称 | 关键产出 | 依赖 |
|------|------|----------|------|
| **A** | 树状 RAG 与导航流程 | 建树、遍历、评测脚本；与 reader 对接 | 数据与 embedding 管线 |
| **B** | 检索头：发现与分析 | 探针、层/头报告、与检索行为相关性 | 固定主干与小任务数据 |
| **C** | 检索头：注入与训练 | 模块 + 训练配置 + 消融表 | A 的稳定流水线、48G |
| **X** | 横切 | Smoke、扫参、环境锁定、回溯协议原型、registry | 无 |

**依赖顺序（逻辑上）**：X → A 验证 harness →（并行）B 分析与 A 真数据 → C 注入 → 论文整合。Mamba 接入优先插在 **A 的 reader 槽位**，与 B/C 可部分并行。

---

## 4. 阶段与时间线（建议）

| 阶段 | 时间（约） | 目标 | 完成标志 |
|------|------------|------|----------|
| **0 基建** | 第 1–2 周 | 双机环境、Git、smoke、实验登记规范 | `scripts/smoke/smoke_local.py` + lock 文件 + registry |
| **1 系统验证（玩具→文本形→扫参）** | 第 2–5 周 | 证明「树 × reader 类型」在效率上有可写差异或明确无差异 | 曲线/表 + `docs/experiments/planning/PHASE1_VALIDATION_PLAN.md` 结论段 |
| **2 真数据浅层树** | 第 4–8 周 | 小语料 RAPTOR 式或层次聚类树 + 同一 harness | 可复现建树脚本 + 1 个 QA/导航任务指标 |
| **3 Mamba-2 接入** | 第 6–10 周 | 第三套 reader；与 TF/GRU(占位) 同网格对比 | 同 CSV 列规范 + registry |
| **4 检索头 B** | 第 8–12 周 | 探针与报告 | 内部技术报告一节可进论文 |
| **5 检索头 C + 联合** | 第 10–14 周 | 注入训练与消融 | 主表 + 附录 |
| **6 扩展与成文** | 第 14–24 周 | 回溯实验、讲故事、投稿准备 | 论文初稿 + 开源复现包 |

*注：周次重叠表示可并行；以你导师节点与会议 deadline 为准裁剪。*

---

## 5. 里程碑产出物

- **工程**：`src/rag_tree/` 建树与 reader 基准；`src/retrieval_head/`；配置与脚本；`results/metrics/*.csv`。  
- **文档**：本文件；**分层索引与「单一权威」矩阵**：**`docs/README.md`**；**当前迭代勾选**：**`docs/overview/execution/CURRENT_SPRINT.md`**；**周历模板**（勿与 sprint 双写）：**`docs/overview/planning/ROADMAP.md`**；**`docs/experiments/planning/PHASE1_VALIDATION_PLAN.md`**；**`docs/experiments/planning/EXPERIMENT_REGISTRY.md`**。  
- **学术**：1 篇主投（系统 + 机制）或 1 系统 + 1 机制短文；具体在阶段 1 结束后根据数据定题。

---

## 6. 实验与数据策略

- **玩具**：已具备（平衡 k 叉树 + 扫参）。  
- **文本形**：短文档切块 + 层次合并（不必一开始上 7B 编码器；可先用确定性 embedding 或小型 encoder）。  
- **真数据**：先 1 个公开小集合（如领域 PDF/维基子集），控制总 token，便于 AutoDL 与本地一致复现。  
- **记录**：每次实验 `git_sha`、`gpu_name`、`torch` 版本写入 CSV（扫参脚本已支持）；registry 一行对应可复现命令。

---

## 7. 基础设施

- **环境**：本地 `conda env mamba2`；5060 需 **PyTorch cu128**（见 `environment/MAMBA2.md`）。  
- **同步**：代码 `git pull/push`；数据/权重网盘或 AutoDL 盘；路径用 `MAMBA2_*` 环境变量。  
- **CI**：可选后续加「仅 CPU 的 import + 极小 forward」；非必须。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Mamba/Windows 编译与 CUDA 不匹配 | 主训放 AutoDL；本地只做 CPU/小模型或 WSL |
| 玩具实验与真任务结论不一致 | 尽早接小真数据；玩具只作假设筛选 |
| 检索头信号弱 | 缩小任务到「是/否检索」二分类 + 更强探针 |
| 时间不够 | 砍掉阶段 6 扩展与 **B-S3**；**优先保留** 主对比 + **SSGS/M1 脚注**；检索探针 **压缩为附录一段** |

---

## 9. 文档与仓库导航

**完整分层索引与「单一权威」**（避免与各 overview 双写长表）：**`docs/README.md`**。

---

## 10. 当前进度快照（维护方式）

**唯一滚动维护**：**`docs/overview/execution/CURRENT_SPRINT.md`**。本文件为 **月级规划**，不在此重复 sprint 勾选。

**结题逻辑序（阶段 0–7）**：与 §4 周次模板 **对照表** 见 **`docs/overview/planning/RESEARCH_PHASES_0_TO_DONE.md`**（**当前主瓶颈 = 阶段 5 投稿包/成文**）。

---

## 11. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | **§1.0**：**主验证轴** = **树状 RAG + Mamba path reader + SSGS/M1**；**副线** = 检索头与探针（B-S2/B-S2+） |
