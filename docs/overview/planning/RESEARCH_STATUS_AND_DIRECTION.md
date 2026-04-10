# 研究现状与方向总览

> **读法**：先读本文 **§1–§3** 建立共识；**§1.5** 为 **长期北星（状态快照回溯 × 树状 RAG）**，与 **阶段 1–2 地基** 的显式分界；**§3.5** 为 **对外讨论（投资/Agent 叙事）的批判性接收与证据层级**；**§4** 为决策规则；**§5** 为推荐执行顺序（与 **`docs/overview/execution/NEXT_RESEARCH_PLAN.md`** 细节一致）。  
> **更长规划**仍以 **`docs/overview/planning/PROJECT_MASTER_PLAN.md`** 为准；**阶段 1 成稿**见 **`docs/experiments/phases/PHASE1_MANUSCRIPT.md`**。  
> **可执行勾选清单**（成文 / 仓库 / 可选实验）：**`docs/overview/execution/NEXT_RESEARCH_PLAN.md`** 篇首 **「当前收口清单」**；**归档路径核对表**：**`PHASE1_MANUSCRIPT.md` §5.1**（同目录 **`docs/experiments/phases/`**）；**本机 5060**：**`docs/environment/runbooks/LOCAL_5060_RUNBOOK.md`**（**`docs/environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md` §11**）。

---

## 1. 整体研究方向（不变）

**论文级主叙事、基线对照表（A/B/C/D）、贡献候选长文**：**唯一权威** **`PROJECT_MASTER_PLAN.md` §1**（本节不复制）。

**执行摘要**：**主对比轴**为 **树内、同 harness** 的 **Mamba vs Transformer（+GRU）**；**平面 RAG / 混合架构** 为消融或讨论。**三条贡献候选**（系统 path-batch 实现敏感；检索头机制；快照协议）的**展开与边界**仍见 **MASTER_PLAN**；**与当前证据的对应**见本文 **§2–§3**。

**主验证轴 vs 副线（与 `PROJECT_MASTER_PLAN.md` §1.0 一致）**：**最终实际验证方向** 收束为 **树状 RAG + Mamba path reader + SSGS（及 M1 同树 DFS 对照）** — 主文主图、Wikitext 扩展、**SSGS/M1 分列脚注（七轴）**。**检索头与 B-S2/B-S2+ 探针** 为 **副线**（附录/机制节），**不**替代上述主叙事；**B-S3/注入** 按需。

### 1.5 长期北星：状态快照回溯与树状 RAG（**超越阶段 1–2**）

> **定位**：本节固化 **最终要讲清的技术内核** 与 **对外动机**；**不**替代 **§3.5** 的证据层级与止损规则。  
> **与阶段 1–2 的关系**：当前仓内 **path-batch / §7 / SSGS / M1 / L3 轨迹** 等，是在 **可发表、可复现** 的前提下，为这一北星 **铺地基**（**L1–L3** 量纲）；**不是**「做完阶段 2 就等于证成了 Agent 级无损撤回」。

**概念（与 Mamba / SSM 一致）**：递推 **\(h_t = f(h_{t-1}, x_t)\)** 下，**\(h_t\)** 是对过往历史的 **固定维压缩**。**快照** = 在某时刻保存 **\(h\)**；**回溯** = 将保存的 **\(h\)** 写回运行态，使解码从该时刻继续。这与 **Transformer 依赖随序列增长的 KV Cache** 在 **状态载体几何** 上根本不同：**一方是固定大小状态向量，一方是随上下文线性膨胀的 KV 张量**。

**树状 RAG 中的对照直觉（叙事用；主文须脚注分列 harness）**：

- **KV 模式（Transformer 系 path reader 的常见代价结构）**：误入子树会 **堆叠** 与已读前缀相关的 KV；回溯往往对应 **丢弃缓存（丢前文可复用性）**、**重算**，或 **显存与碎片管理** 压力。  
- **隐状态模式（Mamba + 显式快照协议，如 SSGS）**：在分支点保存 **\(h\)** 的 **clone**；误入后可用 **O(1) 量级的状态写回**（相对「整段 KV 体量」）回到分支点，再试其它子树。本仓 **SSGS**（`dfs_ssgs_mamba` + `DynamicCache`）与 **M1**（同树 DFS 上对照 **玩具 TF-KV** 的 clone / truncate）都是在 **不同测量轴** 上把这一 **代价结构差异** **做成数字**；**§7** 则提供 **机制毫秒/字节** 的分解尺。

**算法叙事名（贡献包装）**：**State-Snapshot Guided Search（SSGS）** — 在节点完成编码后 **挂载快照**，探索失败则 **加载父快照** 再选兄弟分支；控制流可与 **规则 DFS** 或未来 **可学习策略** 分层讨论（后者 **≠** 当前主文已解决，见 **§3.5**）。

**Agent / 智能体动机（允许写进引言，但必须降维到证据）**：多跳检索里 **错误分支污染上下文** 是常见失效模式；**固定大小状态 + 显式回溯** 提供一种 **「悔棋」式控制流假设** —— 与 **§3.5** 中的 **L4** 叙事同档，**须单独 harness**，**禁止**从 **L1–L2 效率曲线** 直接跳写 **「已解决 Agent 记忆」**。

**价值判断（与 §3.5「批判性接收」一致）**：该方向 **赛道新、与架构特性耦合深**，若能在 **限定任务** 上把 **L3 语义保真** 或 **可学习回退** 做实，**学术上极具张力**；同时 **有损状态、谁触发回退、CPU↔GPU 搬运** 等 **硬风险** 必须进 **讨论/局限**，**不以修辞替代实验**。外部 **「天使轮 / 高收益·中高风险」** 类比喻 **仅用于** 选题沟通，**不**写入可审稿的 **结果句**。

**验证路径**：**§3.5** 已给出 **极简树 + 硬编码读序 + 轨迹对照** 的 **低成本 PoC** 与 **止损规则**；仓库 **阶段 C** **`tf_kv_trajectory_l3_minimal`** 已在 **玩具 TF-KV** 上给出 **错枝/恢复 vs 直达** 的 **表示层** 读数（**≠** 全模型 Agent 证明）。

---

## 2. 当前现状（截至文档更新日）

### 2.1 已完成（可写进论文「系统验证 / 阶段 1」）

| 域 | 状态 | 索引 |
|----|------|------|
| **path-batch 主文** | 3090 **同机** naive vs fused；CSV + 三张主图；结论：**Mamba 峰值强烈依赖 fused** | 登记 **A-20260408-paper-main-3090-***；数据 **`results/metrics_result/paper_main_*.csv`**；图 **`results/metrics/figures/mamba_3090_naive_vs_fused_*.png`** |
| **真语料浅树（效率）** | Wikitext-2 叶块 + **同一 reader harness** | **A-20260408-wikitext-3090-fused**；JSON 在 **`metrics_result/`** |
| **阶段 2 本地试跑（5060）** | Wikitext **`n∈{8,16}` × `chunk_len∈{8,12}`** 共 **4** 点，`WARMUP=2` `REPS=5`；**HF naive** Mamba | **`benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json`**（**`PHASE2_DRAFT.md` §1.1** 表）；旧名 **`benchmark_wikitext_local5060_n16_*`** 仍可对照；**禁止与 3090 fused 混表**（见 **§6**） |
| **阶段 2 A2-S2（3090 fused）** | 与上 **同拓扑四格**，**`WARMUP=2` `REPS=8`**；**mamba_ssm** 融合路径 | **R1** **`…stage2_fused_20260409T1035Z_*`** + **R2** **`…stage2_fused_r2_20260409T1110Z_*`**（峰值一致）；登记 **A-stage2-wikitext-grid-v1** |
| **阶段 2 扩维（dim256）** | 同拓扑四格 **`dim=256`** **fused** | **`benchmark_wikitext_stage2_dim256_20260409T1137Z_*`** + grid CSV；登记 **A-stage2-wikitext-dim256-v1** |
| **阶段 2 大叶数单点（3090）** | **32 叶**、**dim128**、**c8** **fused** | **`benchmark_wikitext_fused_n32_c8_*.json`**；登记 **A-stage2-wikitext-n32-c8-3090-v1** |
| **阶段 2 叶数扫描（3090 fused）** | **固定 c8 dim128**，**`n∈{8,16,32,64}`**；**`WARMUP=2` `REPS=8`** | **`benchmark_wikitext_stage2_leavescale_20260409T1257Z_*`** + **`…_grid_20260409T1257Z.csv`**；登记 **A-stage2-wikitext-leavescale-v1** |
| **阶段 2 叶数 XL（128/256 叶）** | **同上 harness**，**`TAG=stage2_leavescale_xl`** | **n128** **`20260409T1322Z`**、**n256** **`20260409T1324Z`**；**`benchmark_wikitext_stage2_leavescale_xl_*`** + **`…_grid_n128_n256_combined.csv`**；登记 **A-stage2-wikitext-leavescale-xl-v1** |
| **§7 depth 5–6（S1–S4）** | **单路径玩具协议**，与 path-batch **分列** | **`STAMP=20260409T1341Z`**；**`stage2_leavescale_xl_s{1..4}_*_d{5,6}_*.json`**（前缀因 **TAG** 残留误用，见 **登记册脚注**）+ manifest；登记 **X-section7-depth-extension-v1** |
| **§7 玩具协议 S1–S4** | 3090 CUDA 归档 + **串行复跑**通过；与 **§7.3.1** 同阶 | **X-20260421-***；`**_20260421.json`** + **`*_20260408T1617Z.json`** |
| **SSGS × Mamba** | DFS + `DynamicCache` 导航环可复现 | **X-20260421-ssgs-mamba-dfs-demo** |
| **SSGS × Mamba（Wikitext 同树）** | 与 **`benchmark_wikitext_tree`** **同建树** 上 **`dfs_ssgs_mamba`**；**path-batch** 与 **SSGS** 桥接；**叶数** **n∈{8,16,32,64}** **CUDA** + **n8 CPU** 已进 **`ssgs_mamba_wikitext_grid.csv`** | **`demo_ssgs_mamba_wikitext.py`** + **`aggregate_ssgs_mamba_wikitext_json.py`**；登记 **X-20260407-ssgs-mamba-wikitext-tree**；**`tests/test_ssgs_mamba_wikitext.py`** |
| **叙事边界** | 主图 / §7 / SSGS / **M1 DFS** / **L3 轨迹玩具** / 真 LM / **阶段 2 任务** **七条线不混读**（测量轴见 **§3**） | **`FIGURE_CAPTIONS_STAGE1.md`**（篇首 **P0** + **七条测量轴** 表）、`**RESEARCH_NOTES**` §7.0；任务细节 **`PHASE2_DRAFT.md`**（与 **`PHASE1_MANUSCRIPT` §8** 并行） |
| **阶段 1 成文** | 可贴正文的一节（含摘要/方法/结果/§7 关系/归档索引） | **`PHASE1_MANUSCRIPT.md`** |
| **阶段 2 成文（P1）** | **真语料动机 + A2-S3 协议 + 公平性** 已并入主稿 **§8**；**检索头 / Mamba 讨论边界** 见 **§9**；指针 **§10** | **`PHASE1_MANUSCRIPT.md` §8–§10**；**`RETRIEVAL_HEAD_NOTES.md` §8** |
| **阶段 2 任务指标（A2-S3 v0）** | Wikitext 浅树 + **叶对同 cohort** 二分类（**ridge / concat 池化**）；与 path-batch **分列**，**非**墙钟 | **`task_wikitext_path_pair.py`**（**stratified** 或 **leaf_heldout**）；登记 **A-20260407-stage2-wikitext-path-pair**；含 **`…_leafheldout4_{cpu,cuda5060}.json`** 等 |
| **A2-S3 init×5（3090 CUDA）** | **leaf_heldout H=6**、**sibling**、**c8 dim128**、**`init_seed∈{0..4}`**，**n∈{16,32}**；**test 15 叶对** | **`results/metrics_result/task_wikitext_sibling{16,32}_c8_leafheldout6_initseed*_20260409T1438Z.json`**（**10** 文件）；登记 **A-stage2-wikitext-path-pair-initseed5-3090-v1**；**`aggregate_task_wikitext_path_pair_json.py`** |

### 2.2 已完成但定位为「辅线」（默认不增投）

| 域 | 说明 |
|----|------|
| **真 LM 玩具线** | **X-20260422–25**：最小 CE、启发式导航、goal 子头、**SSGS×LM** 并列；**reach_rate&lt;1**，**不**支撑「已解决导航」 |
| **大叶数研究扫参** | **A-20260408-research-large-leaves-3090**；扩展网格另开 TAG |

### 2.3 尚未完成（与主叙事的关系）

| 项 | 与主叙事的关系 |
|----|----------------|
| **阶段 2：真语料 + 任务指标** | **5060 CUDA 四格**、**3090 fused 四格（A2-S2）**、**dim256**、**叶数 8–64 / XL 128–256**、**§7 depth 5–6**、**A2-S3 init×5（3090）** 均已登记（**A2-S3 init×5** JSON：**`metrics_result/*_20260409T1438Z.json`**）；**成文** 见 **`PHASE1_MANUSCRIPT` §8–§9** |
| **检索头 B（分析）** | 支撑贡献候选 **②**；**B-S2**（GPT-2 岭探针 + topic heldout）+ **`RETRIEVAL_HEAD_NOTES` §2 / §5**；**B-S2+**（**`probe_path_reader_linear`**：16 叶 heldout、可选 BCE / **`--train-head-only`**）已本地归档；**per-head / 大模型** 仍待 **B-S3** 与机时 |
| **检索头 C（注入训练）** | 依赖 B 与稳定 harness；**48G** |
| **§7.5 S5 汇总表** | 支撑贡献候选 **③** 的「一句话表」；**可做可不做**，视篇幅 |
| **平面 RAG 消融** | 动机/完整性；**非当前阻塞** |
| **SSGS × KV/重算「同树同任务」harness** | **M1**：**三臂** JSON + **L3**（隐状态 / 固定叶头 CE）；登记 **X-ssgs-vs-kv-tree-nav-m1**；见 **`SSGS_MAINLINE_M1.md`** |

### 2.4 工具就绪（**Phase M1** 正式开工）

**已齐**：`dfs_ssgs_mamba`、`demo_ssgs_mamba_{dfs,wikitext}`、`aggregate_ssgs_mamba_wikitext_json.py`、`run_ssgs_mamba_wikitext_cuda.sh`、§7 **Mamba 快照/恢复** 与 **TF-KV**（含 **`--branch-truncate-demo`**）、相关 **`tests/test_ssgs*.py`**。  
**已落地（M1）**：**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** 输出 **`kind=ssgs_vs_kv_tree_nav_wikitext`**（**SSGS Mamba** + **玩具 TF-KV** clone / truncate_kv）；登记 **X-ssgs-vs-kv-tree-nav-m1**；汇总 **`ssgs_vs_kv_wikitext_nav_grid.csv`**。详见 **`SSGS_MAINLINE_M1.md`**。

---

## 3. 测量轴（防混读；七条）

| 轴 | 回答什么 | 代表登记 / 文件 |
|----|----------|-----------------|
| **Path-batch 主图** | 固定路径集合上 **三 reader 批量前向** 的 **时间与 m2_peak** | **A-20260408-paper-main-3090-pair** |
| **§7 玩具表** | **单路径**上 **clone / restore / TF-R1 / TF-KV** 等 **分列毫秒** | **X-20260421-*** |
| **SSGS demo** | **DFS 试错序** + **token 步进** + cache 快照 | **X-20260421-ssgs-mamba-dfs-demo**；**Wikitext 同树** **X-20260407**（**`ssgs_mamba_wikitext_grid.csv`**） |
| **M1 同树 DFS** | **同 Wikitext 建树**上 **SSGS** vs **玩具 TF-KV**（clone / truncate_kv）**同一 DFS**；**wall_s / peak**；可选 **L3**（隐状态、固定叶头 CE） | **X-ssgs-vs-kv-tree-nav-m1**；**`ssgs_vs_kv_tree_nav_wikitext_*.json`**；**`ssgs_vs_kv_wikitext_nav_grid.csv`** |
| **L3 轨迹（玩具 TF-KV）** | **硬编码** 错枝 **+ restore** **vs** 金路径直达；**末 hidden 余弦**；**≠ M1 DFS** | **X-20260411-tf-kv-trajectory-l3-minimal**；**`tf_kv_trajectory_l3_minimal_*.json`**；**`src/rag_tree/tf_kv_trajectory_l3.py`** |
| **真 LM 线** | **tiny-gpt2** 上 **CE / 导航指标**；**非** path-batch harness | **X-20260422–25** |
| **阶段 2 任务（A2-S3）** | 同 Wikitext 树上的 **效果 proxy**（例：叶对 cohort **ridge 准确率**）；**非**主图纵轴、**非** §7 毫秒 | **A-20260407-stage2-wikitext-path-pair**；**`PHASE2_DRAFT.md`**；成文并入 **`PHASE1_MANUSCRIPT.md` §8–§9** |

**规则**：正文 **禁止**把 §7 某一列与主图纵轴 **当作同一物理「一步」** 相减或混谈。

### 3.5 对外叙事：批判性接收与证据层级（2026-04）

外部讨论常使用 **「高收益/高风险」「天使轮式选题」「Agent 无损撤回」** 等修辞来刻画 **问题重要性**。这些说法可用于 **动机与 Related Work**，但 **不构成** 审稿意义上的证据；本文档 **主结论** 仍以 **登记实验**（**`EXPERIMENT_REGISTRY.md`**）与 **`PHASE1_MANUSCRIPT.md`** 为准。

**如何批判性接收**

- **保留**：问题在 **树导航 / 试错 / 状态可携带** 上与 **KV 堆叠范式** 的 **代价结构差异** 值得研究；**Mamba/SSM** 与 **显式快照** 在工程上 **可测量**（本仓已有 path-batch、§7、SSGS 多轴）。  
- **拒绝**：把 **动机修辞** 直接写成 **已证成的系统结论**（例如「颠覆 RAG」「必中顶会」「状态无损」），除非有 **同层级的对照实验**。

**证据层级（由弱到强；与主文七轴正交，勿跳级混谈）**

| 层级 | 回答什么 | 本仓接近什么 | 尚未作为主线承诺的 |
|------|----------|--------------|---------------------|
| **L1 机制可行** | 状态能否 **clone / restore**、字节与毫秒量级 | §7 玩具协议（**X-20260421-***）、**`benchmark_mamba2_cache_*`** | — |
| **L2 效率是否划算** | 相对重算 / 其它 reader，峰值与墙钟 | **path-batch** 主文、Wikitext 阶段 2、5060 naive **动机**（分列） | 大规模在线服务 SLA |
| **L3 语义是否保真** | 回退后 **表示/输出** 是否仍与「停在节点 A」一致；错枝是否 **污染** 可逆 | **部分**：**M1** 玩具 TF-KV 上 **隐状态余弦 + 固定叶头 CE**（**`l3_tf_kv_*`**）；A2-S3、B-S2+ 为 **proxy**；**非** 端到端「撤回键」证明 | **训练型头 / 树 LM 对齐** 须另 **harness**；**显式对比** 叙事见 **M1** 脚注 |
| **L4 系统级 Agent** | 在线纠错、RL、长程任务 | **未做**；**X-20260422–25** 为 **辅线玩具** | 需另立 harness 与预算 |

**须提前写进「风险」段落的三个问题**（与外部讨论一致；**不**因叙事而消失）

1. **有损压缩**：SSM 隐状态 **固定大小**；长序列下 **细节遗忘** 文献已有。→ **快照 ≠ 无损存档**；结论须限定 **序列长度 / 任务**。  
2. **谁触发回退**：**规则 DFS**（SSGS）与 **可学习策略** 不同；RL/奖励塑造 **难**，**不**假装已在主文解决。  
3. **CPU↔GPU / 显存ↈ内存**：频繁 **H2D/D2H** 可能吃掉 **理论上的 asymptotic 优势**。→ §7 已有 **fromcpu restore** 等线索；**大规模树** 须单独测。

**最小 L3 验证（建议 1–2 周窗口；与「再扫一格 path-batch」二选一投时间）**

- **数据**：**2 层深、≤10 终端叶** 的 **手工微型树**（或复用现有 **浅树** 的最小子图）。  
- **控制流**：**硬编码** 读序；**不**先上大系统（RAPTOR 级）。  
- **核心对照**（示例）：在 **同一解码目标** 下比较  
  - **轨迹甲**：读 **A → 错枝 B → 将 A 的快照 load 回 → 读正枝 C**  
  - **轨迹乙**：读 **A → 直接读 C**  
  报告 **延迟**（是否打断图）+ **任务指标**（如 next-token / 探针 / 简单分类），**登记为新 kind**，**禁止**与 path-batch **无脚注合并**。  
- **止损规则**：若 **restore 后指标系统性崩坏**（「失忆/胡言」），则 **收紧主文声称** 或 **转向纯效率+机制** 叙事，**不**强行升维到 L4。

**仓库实现（阶段 C）**：**`src/rag_tree/tf_kv_trajectory_l3.py`** + **`scripts/research/benchmark_tf_kv_trajectory_l3_minimal.py`**（**`kind=tf_kv_trajectory_l3_minimal`**）；**`pytest tests/test_tf_kv_trajectory_l3.py`**（须 **torch** 可用环境）。登记 **X-20260411-tf-kv-trajectory-l3-minimal**。**已归档 CUDA**：**`results/metrics_result/tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`**（**`cosine_hidden_a_vs_b`≈1** 两臂，**`git_sha=6fa7873`**）。

**与 §4 的衔接**：**L1–L2** 本仓已较充实；若目标是 **顶会级「新算法空间」叙事**，须在 **P0 成文** 之外 **显式排期 L3**；若 **L3 失败**，仍保留 **实现敏感性 + 真语料 harness** 的 **可发表** 主线（见 **§6.1**）。

---

## 4. 决策原则（再决定「先做什么」）

1. **主故事优先**：阶段 1 **效率与 harness** 已齐；**叙事升级** 依赖 **M1**：**SSGS 快照回溯 vs KV/重算** 在 **同一树任务** 上的 **对照证据**（**`SSGS_MAINLINE_M1.md`**）。**检索头 B** 仍为 **辅线**，**不替代 M1**。  
2. **Harness 不拆**：新实验 **必须** 能声明 **与 `benchmark_*_tree` / `run_tree_reader_benchmark` 同槽位**，否则 **新开登记行** 并 **禁止与主表混点**。  
3. **算力分档**：5060 做 **smoke 与脚本**；3090/48G 做 **登记级数字**；**跨机数字** 只作趋势，不作主表格点。  
4. **辅线节流**：**X-20260422–25**、子头 **B（抬 reach）** 仅在有 **明确审稿/假设** 时加投。  
5. **成文并行**：**方法/相关工作** 可与 **A2-S1 smoke** 同周推进，避免等「所有实验完」才动笔。

---

## 5. 推荐执行顺序（已按依赖排序）

下列顺序在 **`NEXT_RESEARCH_PLAN.md`** 中展开；**阶段 2 网格与 A2-S3** 已多行归档。**当前增量主轨** 为 **M1**（见 **`SSGS_MAINLINE_M1.md`**）。

| 顺序 | 动作 | 目的 |
|------|------|------|
| **0** | **M1**：**已登记** **`X-ssgs-vs-kv-tree-nav-m1`**；**n64**、**L3**（隐状态 **n8** + 下游 CE **n8–n64**）、**网格 CSV**（**数据行数 = 聚合脚本 `N row(s)`**，**`DATA_ARCHIVE_202604_SERVER.md`**）已归档；**成文** 固化 **第七轴（M1）**；可选 **`git pull` 后单点 smoke 刷新 `git_sha`** | **玩具轨迹甲·乙** 已 **`tf_kv_trajectory_l3_minimal`**（**第七轴旁支 / 脚注分列**）；**训练型 L3** 另 **`kind`**；**训练型子头** 与树 LM **分列** |
| **1** | **登记**：**EXPERIMENT_REGISTRY** 新增 **阶段 2 / Wikitext 网格** 占位行（如 **`A-stage2-wikitext-grid-v1`**） | 固定 **承诺** 与 **指标列模板**，避免散跑 |
| **2** | **A2-S1**：`benchmark_wikitext_tree.py` **smoke** → JSON 入 **`results/metrics_result/`** | 验证 **HF + fused + harness** 在阶段 2 网格上仍 **可跑通** |
| **3** | **B-S1 / B-S2 / B-S2+**：**`RETRIEVAL_HEAD_NOTES.md`**（§2 / §4 GPT-2 探针；§5 叙事；§7 **path reader** 探针） | **机制线** 与 **系统线** 对齐；**per-head** 仍属 **B-S3** |
| **4** | **A2-S2**：AutoDL **小网格**（2–3 个配置）→ 更新登记 **指标** | 真语料上 **效率曲线/表** 有 **多点** |
| **5** | **A2-S3**：任务指标（**v0**：**`task_wikitext_path_pair.py`** 已落地；cloze / 检索等仍可选） | 主文出现 **非纯延迟** 的 **一行结果**；与 **4** 分列 |
| **6** | **成文**：**阶段 2 半页**（可 **`PHASE2_DRAFT.md`** 或接 **MANUSCRIPT**） | **投稿级** 结构闭环 |
| **7** | **可选**：主图 **PNG 入仓**；**S5** 表；**平面 RAG** smoke | **篇幅与审稿反馈** 驱动 |
| **4b** | **`run_server_wikitext_leavescale.sh`** + **`run_server_section7_depth_sweep.sh`**（**`SERVER_SWEEP_RUNBOOK` §2f–§2g**） | **Wikitext** 叶数 **{8,16,32,64}** 与 §7 **S1–S4** **depth 5–6**；登记 **A-stage2-wikitext-leavescale-v1**、**X-section7-depth-extension-v1** |

**并行**：**2** 与 **3** 可同周；**4** 建议在 **2** 绿之后；**5（A2-S3 v0）** 已与 **2** 同 harness 落地（叶对 cohort 标签），**cloze / 检索** 等仍可选；**4** 与 **5** 正文 **分列**。

---

## 6. 实验公平性与何时换更大模型

### 6.1 当前设计「公平」在哪里

| 对比 | 是否公平 | 说明 |
|------|----------|------|
| **Mamba2 vs Transformer vs GRU** 在 **path-batch harness** | **在槽位内公平** | 同一批路径、同一 `dim/chunk/depth`、同一计时定义（`warmup/reps`、`m2_peak_mib` 口径一致）。比的是 **路径编码器架构 + 实现**，不是比「谁参数更多」。 |
| **naive vs fused Mamba** | **公平（若同机同网格）** | 主文 **A-20260408-paper-main-3090-pair** 即 **同 GPU、同 commit、同计时**；结论应写成 **实现路径敏感**，而非「Mamba 对 Transformer 必胜/必败」。 |
| **5060 CSV vs 3090 CSV** | **不公平作「同一点」** | 只能作 **动机/趋势**；主图必须用 **同机** 数据（见 **`PHASE1_VALIDATION_PLAN.md` §6.2**）。 |
| **path-batch 主图 vs §7 玩具表** | **不可混为同一「一步」** | 各列物理含义不同；**禁止**互减或混谈为同一操作耗时。 |
| **小宽度 path reader vs 全 LLM KV** | **不公平、也不声称公平** | 阶段 1 的 Transformer 是 **小 encoder**；全文须 **分项声明**（已有 **`RESEARCH_NOTES` §7** 与 **`PHASE1_MANUSCRIPT` §3**）。 |

**结论**：在 **「树路径、小 encoder、固定 harness」** 这一 **自洽问题** 上，当前实验 **合理且可发表**；**不能**无限定推广到「7B LLM 端到端 RAG」——除非另开实验与登记。

### 6.2 当前设计「有限」在哪里（需在正文写清）

- **规模**：`dim` 多为 **128–384**，层数 **浅**；结论主要是 **趋势与实现敏感性**，不是 **大模型极限显存** 的绝对值。  
- **任务**：阶段 1 **几乎无任务准确率**；**阶段 2 的 +1 指标** 用来补 **效果** 维度。  
- **数据**：合成 + Wikitext 叶块 **浅树**；**不**等价于大规模工业索引。

### 6.3 什么时候适合换「更大模型」对比

建议 **分档推进**，避免一步跳到 7B 导致 **变量全炸**：

1. **仍在本 harness 内扩宽**（**优先、成本低**）  
   - 增大 **`dim` / `chunk_len` / 层数`**，在 **同机** 上保持 **naive vs fused** 与 **三 reader** 网格；**显存顶满前** 看曲线形状是否 **与阶段 1 一致**（实现敏感是否仍成立）。  
   - **适合时机**：阶段 2 smoke 绿、**AutoDL 可用**、且你希望主文多 **1–2 张「更大宽度」** 附图。

2. **换「更大但仍可控」的骨干**（**中档**）  
   - 例如 **更大 `Mamba2Config` / 更深 Transformer path encoder**（仍走 `inputs_embeds`），或 **冻结小 LM 只测前向峰值**；须 **新开登记行** 并声明 **与 paper_main_v1 网格不可逐点混表**。  
   - **适合时机**：审稿人质疑「是否仅玩具宽度」；或你需要 **与社区常见 hidden size（如 512/1024）** 对齐 **一页附录**。

3. **端到端大 LM + 树导航**（**高档、另立项**）  
   - 涉及 **KV、长上下文、训练/微调**；**公平性**要重新定义（算力预算、是否冻结、是否同一 tokenizer）。  
   - **适合时机**：阶段 2 **任务指标** 与 **检索头 B** 已有清晰假设；**48G 稳定可用**；且论文故事 **明确升级** 为「系统级」而不仅是 path-reader 微基准。

**原则**：**先在同一叙事下把 harness 推到「更大但仍可比」**，再考虑 **换任务形态**；**AutoDL 忙时** 用 **5060 做登记撰写、smoke、笔记（B-S1）**，**不耽误** 主线逻辑闭环。

---

## 7. 相关文档地图

**分层索引与「哪份文件说了算」**：**`docs/README.md`**（含 **单一权威** 矩阵）。**§1–§6** 为本文独有；**不在此维护第二份** 全库文档表。**2026-04 服务器 JSON / CSV 路径索引**：**`docs/experiments/planning/DATA_ARCHIVE_202604_SERVER.md`**。**阶段 0→结题**（实验 + 成功标准 + **当前阶段 5** 勾选）：**`docs/overview/planning/RESEARCH_PHASES_0_TO_DONE.md`**。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初版：现状 + 方向 + 决策 + 推荐顺序 |
| 2026-04-09 | **§6**：公平性边界 + **何时换更大模型**（三档推进） |
| 2026-04-07 | **§2–§3 / §7**：**P1 成文** 与 **A2-S2 待云端** 在 **§2.1 / §2.3** 分列；测量轴与 **`PHASE1_MANUSCRIPT` §8–§9**、**FIGURE_CAPTIONS** 测量轴表、文档地图对齐 |
| 2026-04-09 | **§2.3**：**A2-S2**（3090 fused 四格）已登记；**`metrics_result/benchmark_wikitext_stage2_fused_*_20260409T1035Z.*`** |
| 2026-04-09 | **§2.1**：**A2-S2 R2** **`stage2_fused_r2` `20260409T1110Z`** 归档；与 **R1** 峰值一致 |
| 2026-04-09 | **§2.1**：**dim256 四格** **`20260409T1137Z`**；**32 叶 c8** 单点；**headcheck** 记 **X-20260409-wikitext-headcheck** |
| 2026-04-07 | **§5**：推荐顺序增 **4b**（**`run_server_wikitext_leavescale.sh`**、**`run_server_section7_depth_sweep.sh`**）；登记占位 **A-stage2-wikitext-leavescale-v1**、**X-section7-depth-extension-v1** |
| 2026-04-09 | **§2.1**：**叶数扫描** 归档 **`STAMP=20260409T1257Z`** **`TAG=stage2_leavescale`**；**`EXPERIMENT_REGISTRY` A-stage2-wikitext-leavescale-v1** 补全指标；**§1** 增 **GitHub TLS 失败 / PyCharm 同步** 说明（**`SERVER_SWEEP_RUNBOOK` §1**） |
| 2026-04-09 | **§2.1**：**128/256 叶 XL** **`stage2_leavescale_xl`** **`1322Z`/`1324Z`**；**`A-stage2-wikitext-leavescale-xl-v1`**；**Mamba2 峰值** **≈282 / 562 MiB** |
| 2026-04-09 | **§2.1**：**§7 depth 5–6** **`1341Z`** 归档；**`X-section7-depth-extension-v1`**；**TAG 残留** 致文件名 **`stage2_leavescale_xl_s*`** — 脚本已改 **`SECTION7_TAG`**；**`RUN_AUTOADL_SECTION7_NOW`** 增 **`unset TAG`** |
| 2026-04-10 | **§3.5 新增**：对外叙事（投资/Agent）**批判性接收**；**L1–L4 证据层级**；三风险；**L3 最小 PoC** 与止损；与 **`NEXT_RESEARCH_PLAN`「后续方向」** 对齐 |
| 2026-04-10 | **§7 文档地图**：**`SUBMISSION_PACK.md`**（**P0 A1–A4**） |
| 2026-04-10 | **§2.3–§2.4**、**§4.1**、**§5**：**Phase M1**（**`SSGS_MAINLINE_M1.md`**）**工具就绪** 与 **双臂 harness 缺口**；**主故事** 增 **M1 优先** |
| 2026-04-11 | **§2.1 表**、**§3**、**§3.5 L3**：**M1 harness 已登记**；测量轴 **五→六**（**M1 DFS**）；**§2.4** **已落地** 叙事 |
| 2026-04-11 | **§5 顺序 0**：**M1 n64 + L3 批次** 已归档；**`DATA_ARCHIVE_202604_SERVER.md`**；**§7** 文档地图增 **数据索引** 指针 |
| 2026-04-11 | **§3** 测量轴 **六→七**（**L3 轨迹**）；**§3.5** 增 **阶段 C** 实现指针 |
| 2026-04-11 | **§3.5**：**L3 轨迹 CUDA JSON** **`20260410T1341Z`** 入仓指针 |
| 2026-04-11 | **§2.1** 叙事边界 **七线**（含 **L3 轨迹玩具**）；**§3** **七轴** 正交表述；**§5 顺序 0** 与 **`NEXT_RESEARCH_PLAN` §0** 对齐 |
| 2026-04-11 | **§1.5**：长期 **北星**（快照回溯 vs KV、SSGS、Agent 动机）与 **阶段 1–2 地基** 分界；对齐 **§3.5** 风险与 PoC |
| 2026-04-11 | **§7 文档地图**：**`RESEARCH_PHASES_0_TO_DONE.md`**（阶段 0–7 + 阶段 5 清单） |
| 2026-04-11 | **§5 顺序 0 / M1**：nav grid **N 数据行**（**`aggregate_*` stdout**；全量 JSON 常 **15**）；与 **`SUBMISSION_PACK` §A2** 对齐 |
| 2026-04-11 | **§1**：**主验证轴**（树+Mamba+SSGS/M1）与 **副线**（检索/探针）定调；指针 **`PROJECT_MASTER_PLAN` §1.0** |
