# 研究现状与方向总览

> **读法**：先读本文 **§1–§3** 建立共识；**§4** 为决策规则；**§5** 为推荐执行顺序（与 **`NEXT_RESEARCH_PLAN.md`** 细节一致）。  
> **更长规划**仍以 **`PROJECT_MASTER_PLAN.md`** 为准；**阶段 1 成稿**见 **`PHASE1_MANUSCRIPT.md`**。  
> **可执行勾选清单**（成文 / 仓库 / 可选实验）：**`NEXT_RESEARCH_PLAN.md`** 篇首 **「当前收口清单」**。

---

## 1. 整体研究方向（不变）

**论文级主叙事**（摘自 **`PROJECT_MASTER_PLAN.md` §1**）：在 **树状结构化检索** 场景下，用 **Mamba-2 类模型** 作 **路径 reader / 导航相关组件**，在 **同一树索引与可比预算** 下与 **Transformer 系 reader** 对照，报告 **效率**（延迟、显存、回溯相关成本）与 **效果**（导航/问答等）；并探索 **状态快照式回溯** 相对 KV/重算的边界。

**主对比轴**：**树内、同 harness** 的 **Mamba vs Transformer（+GRU 占位）**。**平面 RAG**、**混合架构** 为 **消融或讨论**，不替代主骨。

**三条可写贡献候选**（从中选 1–2 条做主文，其余附录）：  
① **系统**：path-batch 上 **实现敏感**（naive vs fused）的效率证据 — **阶段 1 已完成主体**；  
② **机制**：**检索头** 探针与注入 — **未做，属阶段 2–4**；  
③ **协议**：**SSM 快照 vs TF-KV** 等 — **§7 玩具协议 + SSGS demo 已具备素材**，**S5 级「同轨迹」总表** 仍可选。

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
| **叙事边界** | 主图 / §7 / SSGS / 真 LM / **阶段 2 任务** **五线不混读**（测量轴见 **§3**） | **`FIGURE_CAPTIONS_STAGE1.md`**（篇首 **P0** + **五条测量轴** 表）、`**RESEARCH_NOTES**` §7.0；任务细节 **`PHASE2_DRAFT.md`**（与 **`PHASE1_MANUSCRIPT` §8** 并行） |
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

---

## 3. 测量轴（防混读；五条）

| 轴 | 回答什么 | 代表登记 / 文件 |
|----|----------|-----------------|
| **Path-batch 主图** | 固定路径集合上 **三 reader 批量前向** 的 **时间与 m2_peak** | **A-20260408-paper-main-3090-pair** |
| **§7 玩具表** | **单路径**上 **clone / restore / TF-R1 / TF-KV** 等 **分列毫秒** | **X-20260421-*** |
| **SSGS demo** | **DFS 试错序** + **token 步进** + cache 快照 | **X-20260421-ssgs-mamba-dfs-demo**；**Wikitext 同树** **X-20260407**（**grid**：**n8–64** **c8** **dim128**，**`ssgs_mamba_wikitext_grid.csv`**） |
| **真 LM 线** | **tiny-gpt2** 上 **CE / 导航指标**；**非** path-batch harness | **X-20260422–25** |
| **阶段 2 任务（A2-S3）** | 同 Wikitext 树上的 **效果 proxy**（例：叶对 cohort **ridge 准确率**）；**非**主图纵轴、**非** §7 毫秒 | **A-20260407-stage2-wikitext-path-pair**；**`PHASE2_DRAFT.md`**；成文并入 **`PHASE1_MANUSCRIPT.md` §8–§9** |

**规则**：正文 **禁止**把 §7 某一列与主图纵轴 **当作同一物理「一步」** 相减或混谈。

---

## 4. 决策原则（再决定「先做什么」）

1. **主故事优先**：凡直接支撑 **「树内 Mamba vs Transformer + 效率证据」** 的，阶段 1 已齐；**下一步应补「树 + 读者 + 任务」或「检索头机制」**，否则长期停在曲线。  
2. **Harness 不拆**：新实验 **必须** 能声明 **与 `benchmark_*_tree` / `run_tree_reader_benchmark` 同槽位**，否则 **新开登记行** 并 **禁止与主表混点**。  
3. **算力分档**：5060 做 **smoke 与脚本**；3090/48G 做 **登记级数字**；**跨机数字** 只作趋势，不作主表格点。  
4. **辅线节流**：**X-20260422–25**、子头 **B（抬 reach）** 仅在有 **明确审稿/假设** 时加投。  
5. **成文并行**：**方法/相关工作** 可与 **A2-S1 smoke** 同周推进，避免等「所有实验完」才动笔。

---

## 5. 推荐执行顺序（已按依赖排序）

下列顺序在 **`NEXT_RESEARCH_PLAN.md`** 中展开为 **A2-S0…**、**B-S1…** 等里程碑。

| 顺序 | 动作 | 目的 |
|------|------|------|
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

| 文档 | 角色 |
|------|------|
| **`PROJECT_MASTER_PLAN.md`** | 6 个月范围与阶段划分 |
| **`PHASE1_MANUSCRIPT.md`** | 阶段 1 正文素材；**阶段 2**（§8）、**检索头讨论边界**（§9）、指针（§10） |
| **`NEXT_RESEARCH_PLAN.md`** | 阶段 2 / B / X **任务展开** |
| **`CURRENT_SPRINT.md`** | **本周勾选** 与阻塞 |
| **`EXPERIMENT_REGISTRY.md`** | **唯一登记真相源** |
| **`RESEARCH_NOTES.md` §7** | 协议与 SSGS 技术细节 |

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初版：现状 + 方向 + 决策 + 推荐顺序 |
| 2026-04-09 | **§6**：公平性边界 + **何时换更大模型**（三档推进） |
| 2026-04-07 | **§2–§3 / §7**：**P1 成文** 与 **A2-S2 待云端** 在 **§2.1 / §2.3** 分列；测量轴与 **`PHASE1_MANUSCRIPT` §8–§9**、**FIGURE_CAPTIONS** 五轴表、文档地图对齐 |
| 2026-04-09 | **§2.3**：**A2-S2**（3090 fused 四格）已登记；**`metrics_result/benchmark_wikitext_stage2_fused_*_20260409T1035Z.*`** |
| 2026-04-09 | **§2.1**：**A2-S2 R2** **`stage2_fused_r2` `20260409T1110Z`** 归档；与 **R1** 峰值一致 |
| 2026-04-09 | **§2.1**：**dim256 四格** **`20260409T1137Z`**；**32 叶 c8** 单点；**headcheck** 记 **X-20260409-wikitext-headcheck** |
| 2026-04-07 | **§5**：推荐顺序增 **4b**（**`run_server_wikitext_leavescale.sh`**、**`run_server_section7_depth_sweep.sh`**）；登记占位 **A-stage2-wikitext-leavescale-v1**、**X-section7-depth-extension-v1** |
| 2026-04-09 | **§2.1**：**叶数扫描** 归档 **`STAMP=20260409T1257Z`** **`TAG=stage2_leavescale`**；**`EXPERIMENT_REGISTRY` A-stage2-wikitext-leavescale-v1** 补全指标；**§1** 增 **GitHub TLS 失败 / PyCharm 同步** 说明（**`SERVER_SWEEP_RUNBOOK` §1**） |
| 2026-04-09 | **§2.1**：**128/256 叶 XL** **`stage2_leavescale_xl`** **`1322Z`/`1324Z`**；**`A-stage2-wikitext-leavescale-xl-v1`**；**Mamba2 峰值** **≈282 / 562 MiB** |
| 2026-04-09 | **§2.1**：**§7 depth 5–6** **`1341Z`** 归档；**`X-section7-depth-extension-v1`**；**TAG 残留** 致文件名 **`stage2_leavescale_xl_s*`** — 脚本已改 **`SECTION7_TAG`**；**`RUN_AUTOADL_SECTION7_NOW`** 增 **`unset TAG`** |
