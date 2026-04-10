# 阶段 1 成文稿（可直接迁入论文/技术报告）

> **用途**：将阶段 1 **系统验证**整理为连续叙述；数字与路径以 **`EXPERIMENT_REGISTRY.md`** 与本文 **§5 归档索引** 为准。  
> **勿与 §7 玩具协议混读**：path-batch 主结果与 S1–S4 **各列毫秒** 的物理含义不同，见 **`FIGURE_CAPTIONS_STAGE1.md`** 篇首与 **`RESEARCH_NOTES.md` §7.0**。**阶段 2**（5060 动机表、A2-S3 准确率）为 **第五轴**，与主图纵轴 **不可混读**，见 **`FIGURE_CAPTIONS_STAGE1.md`** **「五条测量轴」** 与 **`RESEARCH_STATUS_AND_DIRECTION.md` §3**。

---

## 摘要（约 150 字）

在树状索引上，以 **同一批根—叶路径** 对 **Transformer、GRU、Mamba-2** 三类路径 reader 做批量前向，测量 **步均耗时** 与 **CUDA 峰值显存**。实验表明：**Mamba-2 path reader 的可观测效率对是否启用 `mamba_ssm` 融合实现极为敏感**——无融合核时峰值可达 **GiB** 量级，与 **Transformer/GRU** 的 **百 MiB** 量级形成反差；融合后同网格峰值可降至 **约 51–217 MiB**（随 **叶数与 dim** 可升至 **数百 MiB–约 1 GiB 以下**，仍与 naive **分列**）。故在树形 RAG 的 path-batch 负载下，**不能**仅用 SSM 名义复杂度替代实测效率；主文须 **同机、同 commit** 报告 naive 与 fused 对照。**阶段 2** 在 **Wikitext-2 浅树** 上沿用同一 harness，已归档 **四格、dim256、叶数 8→256** 等 **登记级** 效率曲线；并以 **叶对 cohort + 岭回归** 给出 **与墙钟分列** 的 **效果 proxy**（含 **3090** 上 **五种子** 扫描，**§8.2**）。**§7 玩具协议** 已扩展 **depth 5–6**（**§4**）；**SSGS×Mamba**（DFS + token 步进 + cache 快照）与主图 **非同一 harness**；**Wikitext 同树** 上已归档 **snapshots/rollbacks** 与 **`ssgs_mamba_wikitext_grid.csv`**（**§4**、**§5**）。

---

## 1 问题与范围

**研究问题**：在 **固定建树与遍历协议** 下，**状态式路径编码器（Mamba-2）** 与 **全局自注意力式路径编码器（Transformer）** 及 **递归基线（GRU）** 在 **延迟与显存** 上如何随 **叶数 batch、chunk、深度、宽度** 变化？

**范围**：合成平衡树与 **文本形叶**；reader 为 **小型** 编码器，**非** 全规模 LLM 的 KV 分项测量。**状态快照与 KV 回溯**的对比见 **独立玩具协议**（§7，登记 **X-20260421-***），**不得**与本文 path-batch 主表混为同一「一步」。

---

## 2 方法概要

- **树**：完全 `k` 叉、深度 `d`，叶数 \(k^d\)；节点为 `chunk_len × dim` 嵌入（合成或来自叶文本的确定性嵌入）。  
- **任务**：对给定路径集合 **批量前向**（`run_tree_reader_benchmark` / `sweep_tree_benchmark.py`），记录 `elapsed_s`、`per_step_s`、`m2_peak_mib`（CUDA 上为 `torch.cuda.max_memory_allocated` 在单次基准内的增量峰值）。  
- **主文对照**：在 **AutoDL / RTX 3090** 上，**同一网格、同 `WARMUP`/`REPS`**，分别运行 **fused**（`mamba_ssm` + `causal_conv1d`）与 **`mamba2_naive`（HF 回退）** 环境，得到成对 CSV；用 **`plot_mamba_naive_vs_fused.py`** 生成主图。登记：**A-20260408-paper-main-3090-{fused,naive,pair}**。

---

## 3 主要结果（叙事句）

1. **实现敏感性**：在 **path-batch** 设定下，Mamba-2 的 **峰值显存** 在 **naive** 与 **fused** 之间可出现 **约两个数量级** 的差异；主文结论应明确写 **「融合实现」** 为前提。  
2. **与 Transformer/GRU 的相对位置**：在 **5060 + naive Mamba** 等设定下，Mamba-2 的峰值与步耗可 **显著劣于** 同网格的 Transformer/GRU（动机数据，见本地扫参 CSV）；**3090 主文**以 **同机 pair** 为准。  
3. **语料泛化（浅树）**：在 **Wikitext-2 叶块** 构造的浅树上使用 **同一 reader 槽位**，已从 **小网格** 扩展到 **dim256**、**叶数直至 256**（**§8.4**），支持「不止合成树」且 **规模可扫** 的叙述。  
4. **path-batch 墙钟与峰值的表述边界**：在 **小宽度、真语料 fused** 设定下，**步均耗时** 未必优于 **Transformer/GRU**（micro-benchmark 有抖动）；**大叶数** 时 **Mamba2 峰值可高于** 同 harness 的 **小 encoder Transformer**（例：**256 叶** 档 **约 0.56 GiB vs 约 0.39 GiB**），正文须 **分项写清**，并继续以 **5060 naive GB** 与 **3090 fused** **分列** 支撑 **「实现路径决定可观测峰值」** 的主结论。

### 3.1 阶段 2 本地补充（5060 CUDA，**非主表**）

在 **本机 RTX 5060、HF naive Mamba** 上，对 Wikitext 浅树 **`benchmark_wikitext_tree.py`** 跑 **`n∈{8,16}` × `chunk_len∈{8,12}`** 共 **四格**，`dim=128`，`WARMUP=2`，`REPS=5`。**Mamba2 `peak_alloc_mib`** 约在 **1.1GiB（8 叶）** 与 **2.2GiB（16 叶）** 两档；**`chunk_len` 8↔12** 对峰值 **影响很小**，与阶段 1「叶 batch 主导峰值」的叙述 **一致**。原始 JSON 与 **汇总 CSV** 见 **`results/metrics_result/benchmark_wikitext_5060_cuda_*_20260407.json`**、**`benchmark_wikitext_5060_cuda_grid_20260407.csv`**（**§8.1** 与 **`PHASE2_DRAFT.md` §1.1** 同表）。**禁止**与 **3090 fused** 主文表 **无标注混点**。**效果 proxy**（叶对 cohort、ridge）见 **`task_wikitext_path_pair.py`**、**§8.2** 与 **`PHASE2_DRAFT.md` §2**，与 path-batch **分列**。

---

## 4 与 §7 玩具协议的关系（一段话）

**§7**（S1–S4）在 **单条** 合成路径上分别测量 **Mamba `DynamicCache` clone**、**Transformer 整段重算（TF-R1）**、**带 KV 的增量前向（TF-KV）**、**快照 restore** 等，**各列不可互换**。该协议用于 **机制层与回溯叙事**，**补充** path-batch 主图，**不替代**主图曲线。已在 **CUDA** 上 **串行复跑**（`run_path_protocol_cuda.sh`），新 JSON 与 **`RESEARCH_NOTES` §7.3.1** 及仓内 `*_20260421.json` **同阶**，详见 **`PHASE1_COMPLETE_SUMMARY.md` 附录 B**。

**depth 扩展（登记 X-section7-depth-extension-v1）**：在 **`tree_depth_param ∈ {5,6}`**（路径 **6 / 7** 节点；与 **32 / 64 叶** 同深）上各跑 **S1–S4** 全套，**`STAMP=20260409T1341Z`**。**TF-KV** 末段 **`kv_cache_nbytes`** 在 **d5→d6** 上由 **约 96 KiB → 约 112 KiB** 量级抬升；**S1** **Mamba cache** **clone_nbytes** 仍 **约 41 KiB/段**（与 **depth=4** 归档同阶）。产出文件名曾因 shell **残留 `TAG`** 带 **`stage2_leavescale_xl_`** 前缀，**`manifest` 的 `kind=section7_s1_s4_depth_sweep`** 可据以识别；详见 **`EXPERIMENT_REGISTRY`** 该行脚注。此后脚本改用 **`SECTION7_TAG`**，见 **`RUN_AUTOADL_SECTION7_NOW.md`**。

**SSGS（State-Snapshot Guided Search）与 Mamba cache**：**`dfs_ssgs_mamba`** 在 **DFS 试错序** 下以 **token 步进** 驱动 **HF `Mamba2Model` + `DynamicCache`**，并用 **clone / zero_ / copy_** 做快照与回滚；**玩具树** 演示见 **X-20260421-ssgs-mamba-dfs-demo**。**同一文本 8 叶树** 上 **SSGS 必达** 与 **tiny-gpt2 子头贪心** 的并列指标见 **X-20260425**。**与 `benchmark_wikitext_tree` 同建树** 的 **Wikitext-2** 接线见 **`demo_ssgs_mamba_wikitext.py`**（登记 **X-20260407-ssgs-mamba-wikitext-tree**）：已归档 **`results/metrics_result/ssgs_mamba_wikitext_*.json`** 与汇总 **`ssgs_mamba_wikitext_grid.csv`**（**n∈{8,16,32,64}**，**c8 dim128**、**目标最右叶**；**`snapshots_taken=n−1`**、**`leaf_checks=n`** 可作 **结构性自检**；**`rollbacks`** 随 n **上升**）。**禁止**与 **path-batch** 的 **wall-clock / m2_peak** 或 **§7 单列毫秒** **混为同一纵轴**；可与 **A-stage2-wikitext-leavescale**（同 **c8 dim128** 的 **效率曲线**）**并列叙述、分列子表**。可选续作：**n=128**、或 **`git pull` 后重跑一格** 以刷新 **`git_sha`** — **非**主文阻塞（**§10**）。

---

## 5 数据归档索引（本仓库）

**统一目录**：`results/metrics_result/`（本机示例：`D:\cursor_try\mamba2\results\metrics_result`）。**登记与正文引用时请写相对仓库根的路径。**

| 类别 | 文件名模式 | 说明 |
|------|------------|------|
| **主文 fused CSV** | **`paper_main_dim128_localgrid_paper_main_v1.csv`**；**`paper_main_dim{256,384}_paper_main_v1.csv`** | 3090 fused；**dim128** 文件名为 **localgrid**（与 dim256/384 **同登记轮次**，见 **登记册**） |
| **主文 naive CSV** | **`paper_main_dim128_localgrid_paper_main_naive_v1.csv`**；**`paper_main_dim{256,384}_paper_main_naive_v1.csv`** | 同机 naive |
| **Manifest** | `paper_main_manifest_paper_main_{v1,naive_v1}.txt` | 环境元数据 |
| **主图** | `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png` | 与 **pair** 登记对应（相对仓库根） |
| **§7 复跑（2026-04-08）** | `mamba2_cache_snap_segments_depth4_cuda_20260408T1617Z.json` 等 6 文件 | S1–S4 + branchdemo；另含 `*_20260408T1030Z` 为同协议早次复跑 |
| **Wikitext** | `benchmark_wikitext_3090_fused_20260408T0846Z.json` | 浅树 fused |
| **Wikitext（5060 CUDA，动机）** | `benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json`、`benchmark_wikitext_5060_cuda_grid_20260407.csv` | **HF naive** Mamba **2×2**；**`aggregate_wikitext_5060_cuda_grid.py`** 可重生成 CSV；**不可**与 3090 fused **同表混点** |
| **大叶数研究** | `sweep_research_large_leaves_*_research_lg_v1.csv` + manifest | 登记 **A-20260408-research-large-leaves-3090** |
| **阶段 2 叶数扫描（3090 fused）** | `benchmark_wikitext_stage2_leavescale_20260409T1257Z_n{8,16,32,64}_c8.json`、`…_grid_20260409T1257Z.csv`、manifest | **A-stage2-wikitext-leavescale-v1** |
| **阶段 2 叶数 XL** | `benchmark_wikitext_stage2_leavescale_xl_*_{1322Z_n128,1324Z_n256}_c8.json`、`…_grid_n128_n256_combined.csv` | **A-stage2-wikitext-leavescale-xl-v1** |
| **阶段 2 dim256 四格** | `benchmark_wikitext_stage2_dim256_20260409T1137Z_*` + grid CSV | **A-stage2-wikitext-dim256-v1** |
| **§7 depth 5–6** | `stage2_leavescale_xl_s{1..4}_*_d{5,6}_20260409T1341Z.json`（前缀见 **§4** 脚注）、`…_manifest_20260409T1341Z.txt` | **X-section7-depth-extension-v1** |
| **A2-S3 init×5（3090）** | `task_wikitext_sibling{16,32}_c8_leafheldout6_initseed{0..4}_20260409T1438Z.json` | **A-stage2-wikitext-path-pair-initseed5-3090-v1**；**`aggregate_task_wikitext_path_pair_json.py`** |
| **SSGS × Mamba × Wikitext 同树** | `ssgs_mamba_wikitext_*.json`、`ssgs_mamba_wikitext_grid.csv` | **X-20260407-ssgs-mamba-wikitext-tree**；**`aggregate_ssgs_mamba_wikitext_json.py`**；与 **path-batch**、**§7** **分列**（**§4**） |
| **B-S2+ 本机 5060 CPU** | **`results/metrics/probe_path_reader_linear_text{8,16}_heldout_local5060.json`** | **X-20260407-local5060-bs2plus-rerun**；**`LOCAL_5060_RUNBOOK.md`**；与 **3090 CUDA** **分列** |

**历史归档**（仍在 `results/metrics/`）：`**_20260421.json** 系列，与 **X-20260421-*** 登记一一对应；与 `metrics_result` 中 **STAMP** 文件 **并存**，便于 diff。

### 5.1 成文核对（本仓库路径；**2026-04-07**）

以下已由仓库内 **文件存在性** 核对（投稿前若重跑数据，以 **JSON/`git_sha`** 为准再扫一遍）。

| 核对项 | 状态 | 备注 |
|--------|------|------|
| **主文 CSV + Manifest** | **齐** | **`paper_main_dim128_localgrid_paper_main_{v1,naive_v1}.csv`**；**`paper_main_dim{256,384}_paper_main_{v1,naive_v1}.csv`**；**`paper_main_manifest_paper_main_{v1,naive_v1}.txt`** |
| **主图 PNG（naive vs fused）** | **齐** | **`results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`**（各一张） |
| **§7 + 玩具 JSON** | **齐** | **`metrics_result/`** 内 **`…1617Z`** 等；**`results/metrics/*_20260421.json`** 与登记 **X-20260421-*** 对应 |
| **5060 Wikitext 动机** | **齐** | **`benchmark_wikitext_5060_cuda_*_20260407.json`**、**`…_grid_20260407.csv`** |
| **阶段 2 path-batch 归档** | **齐** | **leavescale / XL / dim256 / §7 depth** 等见 **§5** 表与 **登记册** |
| **A2-S3 init×5** | **齐** | **`task_wikitext_sibling{16,32}_*_1438Z.json`** |
| **SSGS Wikitext** | **齐** | **`ssgs_mamba_wikitext_grid.csv`** + **`ssgs_mamba_wikitext_*.json`** |
| **B-S2+ 本机 5060** | **齐** | **`probe_path_reader_linear_*_local5060.json`**（登记 **X-20260407-local5060-bs2plus-rerun**） |

**四类数字分列脚注（正文须显式）**：**①** **5060 + HF naive** 与 **②** **3090 + fused** **不可同表无标注混点**；**③** **path-batch 墙钟/m2_peak** 与 **④** **§7 单列毫秒**、**⑤** **SSGS 快照/回滚计数**、**⑥** **A2-S3 准确率** **各为独立测量轴**（**`FIGURE_CAPTIONS_STAGE1.md`** 篇首）。**§7.5 S5**「同轨迹总表」仍为 **可选**，视截稿篇幅（**§10** 第 4 条）。

---

## 6 结论段（约 200 字，与 `PHASE1_VALIDATION_PLAN.md` §6.3 一致）

在树路径批量编码设定下（`fanout=2`、`dim=128`、多组 `depth×chunk_len`），HuggingFace **无融合核** 的 Mamba-2 path reader 的 CUDA **峰值显存** 可达 **GiB 级**（例如 64 条并行路径时约 **8.9GiB**），而同设定下 **Transformer / GRU** 路径 reader 多在 **百 MiB** 量级。相对地，在 **AutoDL** 上启用 **`mamba_ssm` 融合实现** 后，**同一网格**上 Mamba2 峰值可降至 **约 51–217MiB**（随配置变化），降幅可达 **两个数量级**。因此：**在树形 RAG 的 path-batch workload 中，可观测效率高度依赖实现是否融合，而不能仅由 SSM 架构名义复杂度替代。** 在 **Wikitext 浅树** 上补充的 **叶对 cohort 岭回归 proxy**（**§8**）**不**改变上述效率主结论，仅提供 **与墙钟分列** 的 **可读性标量**；**§7** 与 **Wikitext 同树 SSGS**（**§4**、**`ssgs_mamba_wikitext_grid.csv`**）则服务 **回溯与导航过程代价**（**快照/回滚计数**），**禁止**与主图 **毫秒/峰值** 或 **§7 玩具毫秒列** 混读。

---

## 7 英文摘要（可选，约 120 词）

We benchmark Transformer, GRU, and Mamba-2 **path readers** on tree-structured retrieval paths under a fixed **path-batch** harness, reporting step time and **CUDA peak memory** (`max_memory_allocated`). On identical grids, **HuggingFace Mamba-2 without fused kernels** can reach **GiB-scale** peaks whereas **Transformer/GRU** stay around **hundreds of MiB**; with **`mamba_ssm` + `causal_conv1d`**, Mamba-2 peaks drop to **tens to hundreds of MiB** on small/medium leaves and remain **sub-GiB** at **larger leaf counts** in our fused runs—still **not** directly comparable to **HF-naive** rows without labeling. Thus, **observed efficiency is implementation-dependent**. A **toy protocol** (§7) measures clone/restore/KV-increment **per column**; **depth 5–6** extensions are archived separately. **Phase 2** adds **Wikitext-2** under the **same reader slot** (efficiency grids through **256 leaves**) and a **ridge-on-concat-pooled-features** **leaf-pair cohort** proxy (**§8**), **not** merged with wall-clock tables. **SSGS×Mamba** (DFS + token steps + `DynamicCache` snapshots) on the **same Wikitext-built tree** is an **auxiliary** line with **snapshot/rollback counts** in **`ssgs_mamba_wikitext_grid.csv`**, **not** wall-clock path-batch; see **§4** and **`FIGURE_CAPTIONS_STAGE1.md`**. **§9**: retrieval-head wording; **§9.1** auxiliary **B-S2+** ridge probes on **synthetic topic leaves** (local CPU), **not** Wikitext A2-S3.

---

## 8 阶段 2：真语料浅树与效果 proxy（正文迁移稿）

本节将 **`PHASE2_DRAFT.md`** 的核心收束进主文素材；**登记与文件名**仍以 **`EXPERIMENT_REGISTRY.md`** 为准。**图注边界**与主图关系见 **`FIGURE_CAPTIONS_STAGE1.md`** 篇首 **P0**（已含 **阶段 2 表** 与 path-batch 的区分）。

### 8.1 系统线：Wikitext 与 path-batch **同 harness**

在 **Wikitext-2 raw** 叶块上 **自底向上** 建树，使用与合成树相同的 **`benchmark_wikitext_tree.py`** 槽位：**Transformer / GRU / Mamba2 path reader**。**登记级 fused 单点**见 **A-20260408-wikitext-3090-fused**（**3090**）。**本地动机（5060 Laptop CUDA，HF naive Mamba）**：固定 **`dim=128`**、**`WARMUP=2`**、**`REPS=5`**，在 **`num_leaves∈{8,16}` × `chunk_len∈{8,12}`** 四格上测量 **`m2_peak_mib`** 与步耗；**Mamba2 峰值**约 **1.1GiB（8 叶）** 与 **2.2GiB（16 叶）** 两档，**`chunk_len` 8↔12** 对峰值 **影响很小**（叶 batch 规模主导）。**原始 JSON** 与 **汇总 CSV**：**`results/metrics_result/benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json`**、**`benchmark_wikitext_5060_cuda_grid_20260407.csv`**（**`scripts/benchmarks/aggregate_wikitext_5060_cuda_grid.py`** 可重生成）。**正文规则**：上述数字 **仅** 标注 **5060、HF naive**；**禁止**与 **3090 fused** 主文表 **无脚注混列**。

### 8.2 任务线（A2-S3）：叶对「同 cohort」二分类

**目的**：在 **同一棵 Wikitext 浅树、同一叶序** 上给出 **可复现的效果 proxy**，避免全文仅有延迟/显存曲线。

**标签**：对叶索引 \(i<j\)，按块大小 \(b\) 定义 **同 cohort**：**\(y=1\)** 当且仅当 \(\lfloor i/b\rfloor=\lfloor j/b\rfloor\)。默认 **`sibling`**：**\(b=\texttt{fanout}\)**；**`root_child`**：**\(b=\texttt{fanout}^{d-1}\)**（详见 **`src/rag_tree/path_pair_geometry.py`**）。

**特征与读出**：对每个叶路径做 path reader，**池化**得 \(z_i\)，拼接 **\([z_i,z_j]\)**，**岭回归** 二分类；并报告 **raw mean-pool 拼接** 基线。**脚本**：**`scripts/research/task_wikitext_path_pair.py`**；**JSON** 字段 **`ridge_concat.*.test_acc`** 等；早期 smoke 见 **`results/metrics/task_wikitext_*.json`**（登记 **A-20260407-stage2-wikitext-path-pair**），**3090 fused** **五种子** 归档见 **`results/metrics_result/task_wikitext_sibling{16,32}_c8_leafheldout6_initseed{0..4}_20260409T1438Z.json`**（**A-stage2-wikitext-path-pair-initseed5-3090-v1**）。

**划分协议**：默认 **`stratified`** 对全体叶对分层抽样 train/test（**`--split-seed`** 只在此模式影响 **哪些叶对** 进 train/test）；推荐 **`--pair-split leaf_heldout --heldout-leaves H`** — train 叶对仅来自 **`[0,n-H)`**，test 仅来自 **`[n-H,n)`**，**避免**同一叶同时出现在 train/test 叶对中（仍对 **全树** 一次前向算嵌入）。**leaf_heldout** 下划分 **完全确定**，多种子试验应扫 **`--init-seed`**（随机化 reader 权重；叶块嵌入仍由文本哈希决定）。**test 叶对数**为 **C(H,2)**，**H** 过小时 ridge **test** 方差大；归档含 **H=4/6** 及 **`chunk_len=12`** 等变体。

**与 §3–§5 的关系**：本任务报告 **准确率类标量**，**不是** path-batch 的 **wall-clock / m2_peak**；正文应 **分列子表或脚注**，**不得**与 **`paper_main_*`** 无标注合并。

**3090、`leaf_heldout` H=6、五种子（`init_seed∈{0..4}`，`STAMP=20260409T1438Z`）**：**test** 仅 **C(6,2)=15** 对，**类别极不平衡**，数值 **不宜过度外推**。用 **`aggregate_task_wikitext_path_pair_json.py`** 对 **n=16** 与 **n=32** 各 **5** 份 JSON 汇总 **`ridge_concat.*.test_acc`** 时，**典型量级**为：**n=32** 下 **GRU** test 均值 **约 0.67**（**std 约 0.16**），**Transformer 约 0.37**，**Mamba2 约 0.19**，**raw baseline 恒 0.2**；**n=16** 下 **GRU 约 0.35**，**Mamba2 约 0.24**。正文可写 **「小样本 held-out 上 GRU 的线性可分性常高于 Mamba2/基线」**，并强调 **任务 v0** 与 **未训练 reader**。

### 8.3 阶段 2 公平性（列语义速查）

| 数字来源 | 正文须标明 |
|----------|------------|
| **5060 CUDA，HF naive** | **naive**、本机 GPU；**非** fused 主文格点 |
| **3090 fused** | **`mamba_ssm` / fused**；登记 **A-20260408-wikitext-3090-fused** 等 |
| **A2-S3 JSON** | **任务名、划分（stratified / leaf_heldout）、`chunk_len`** |

**A2-S2（3090 fused，已归档）**：**`WARMUP=2` `REPS=8`**，**RTX 3090**、驱动 **580.105.08**，**`torch 2.11.0+cu126`**。**R1/R2 四格**（**`STAMP=20260409T1035Z`**、**`20260409T1110Z`**）：**Mamba2 `peak_alloc_mib`** 两轮一致 **约 53 / 55 / 73 / 78**。**扩维 dim256**（**`1137Z`**）：**Mamba2** 四格 **约 62→87 MiB**。**32 叶单点**（**`1140Z`**）：**Mamba2≈98 MiB**。**`git_sha`** 上述跑次在服务器上 **常为 `6fa7873`**（与 **paper_main** 同提交）；与 **仅文档更新的 GitHub HEAD** 可能不一致 — 正文须 **manifest / 登记** 脚注。**5060 naive** 与 **3090 fused** **分列**。**命令**：**`SERVER_SWEEP_RUNBOOK` §2d/§2e**。**叶数扫描、XL、§7 深度** 见 **§8.4**。

### 8.4 叶数扫描、XL 与 §7 深度扩展（登记级摘要）

以下均为 **Wikitext-2 浅树**、**path-batch** **同 harness**（**`benchmark_wikitext_tree.py`**），与 **§7 单路径协议** **分列**（§7 扩展见 **§4**）。**`TransformerPathReader`** 为 **整段 self-attention（O(T²)）**；与 **§7 TF-KV 增量 trunk** **不同对象**，勿混读。

| 登记 id | 内容 | 路径要点（均在 **`results/metrics_result/`**） |
|---------|------|-----------------------------------------------|
| **A-stage2-wikitext-leavescale-v1** | **n∈{8,16,32,64}**，**c8**，**dim128**，**`1257Z`** | **`benchmark_wikitext_stage2_leavescale_*`** + **`…_grid_*.csv`**；**Mamba2 峰值** **约 53→152 MiB** |
| **A-stage2-wikitext-leavescale-xl-v1** | **n∈{128,256}**，**c8**，**dim128**，**`1322Z`/`1324Z`** | **`benchmark_wikitext_stage2_leavescale_xl_*`** + **`…_grid_n128_n256_combined.csv`**；**Mamba2** **约 282 / 562 MiB**；**256 叶** 档 **Mamba2 峰值 > TF/GRU**，须 **正文限定** |
| **X-section7-depth-extension-v1** | **S1–S4**，**depth 5–6**，**`1341Z`** | **`stage2_leavescale_xl_s*_d{5,6}_1341Z.json`**（文件名前缀为 **TAG 残留**，见 **§4**）+ manifest |

**写作提示**：**path-batch** 上 **步均耗时** 不必宣称 **Mamba 全面最快**；**价值**在于 **fused 相对 naive 的量级**、**真语料上随叶数/dim 的曲线**，以及与 **5060 动机** **分列** 的 **公平脚注**。

---

## 9 讨论要点：「检索头」文献与本文探针的边界

近期工作（如 **Retrieval Head**，Yao Fu 等，arXiv:2404.15574）在 **Decoder multi-head self-attention** 上通过 **注意力权重** 识别少数 **头** 在长上下文中的 **信息路由** 角色。**Mamba-2 path reader** 在本 harness 中 **不提供** 与之 **同构** 的 **逐头 QK 注意力图**；配置中的 **`num_heads` / `head_dim`** 属于 **SSD/Mamba-2 块内部** 张量划分，**不可**直接等同于上述文献中的 **「检索头」** 对象。

**本文机制线（B-S2 / B-S2+）** 采取 **更弱但可并列** 的问题：**冻结或小 reader 的池化/层向量上，某类合成标签（如 topic、叶模板）是否线性可读？** — **`probe_retrieval_correlation.py`**（GPT-2 等）与 **`probe_path_reader_linear.py`**（**含 Mamba2 分支**）产出的 **准确率** 是 **表征级** 证据，**不是** **头级因果** 结论。详细文献表与 **Mamba 边界** 见 **`docs/research/RETRIEVAL_HEAD_NOTES.md`**（尤其 **§8**）。

**写作建议**：若讨论中涉及 Mamba，宜用 **「表示中是否含可线性读取的路由/主题信息」** 或 **「层状探针」** 等表述；**避免**写 **「我们发现 Mamba 的检索头」**，除非对 **「头」** 另给 **操作定义** 并单独对照文献。

### 9.1 本机 **5060 CPU** 复跑（B-S2+，合成 **topic** 叶；登记 **X-20260407-local5060-bs2plus-rerun**）

在 **`probe_path_reader_linear.py`** 默认设定下（**STEM / 生活** 各 **n/2** 句叶模板，**`leaf_split=heldout`**，**ridge** 作用于 **未训练** reader 的 **池化向量**），本仓归档：**`results/metrics/probe_path_reader_linear_text8_heldout_local5060.json`**、**`…_text16_heldout_local5060.json`**（**`git_sha`** 以文件内为准）。**`ridge_untrained.*.test_acc`（test）** 摘要：**8 叶** — Transformer **0.5**，GRU **0.25**，Mamba2 **1.0**；**16 叶** — Transformer **1.0**，GRU **0.75**，Mamba2 **1.0**（同 JSON 内 **baseline_raw_mean_pool** 在 **16 叶** 上亦为 **1.0**，表明该 **合成标签 + 确定性叶嵌入** 下 **线性可分性极强**，数值 **不宜外推** 为「真实检索难度」）。

**与 §8.2（Wikitext A2-S3）的关系**：此处为 **path-batch 同型 reader** 上的 **合成 topic 探针**；**A2-S3** 为 **Wikitext 叶块 + 叶对 cohort**。二者 **非同一任务**，正文须 **分列**。**下一步（可选）**：**3090 CUDA** 上同脚本 **去 `--cpu`** 得 **与 5060 CPU 分列** 的一行 JSON；见 **`NEXT_EXPERIMENTS_COMMANDS.md` §6**、**`LOCAL_5060_RUNBOOK.md`**。

---

## 10 下一阶段与文档指针

**本版成文**：**§8.4** 与 **§5** 已收口 **A2-S2 / dim256 / 叶数 8–256 / §7 depth 5–6**；**A2-S3** **3090** **`init_seed`×5**（**n16/n32**、**H=6**）JSON 已入仓 **`results/metrics_result/*_20260409T1438Z.json`**，登记 **A-stage2-wikitext-path-pair-initseed5-3090-v1**。

**主线是否继续？** — **系统主线**（path-batch **效率**、真语料 **Wikitext 扫参**、**§7 玩具协议**、**A2-S3 任务 proxy**）在 **登记级材料** 上 **已闭环**；**不是**弃题，而是 **不必为同一故事无限加格点**。后续工时优先 **把已有数字写进论文/报告**（主文+附录+脚注），其次才是 **可选实验**。

**下一步（按优先级）**：

1. **成文整合（主线收尾）**：把 **`PHASE1_MANUSCRIPT` §8**、**`EXPERIMENT_REGISTRY`**、**`metrics_result`** 中表与 JSON **对齐成投稿版**（含 **五轴图注**、**5060 vs 3090** 分列）。  
2. **可选机制线 B**：**本机 5060 CPU** 已归档 **B-S2+**（**§9.1**，**X-20260407-local5060-bs2plus-rerun**）。若要加强 **与云端对照** — **3090** 上 **1 条** **B-S2+** JSON（**`probe_path_reader_linear.py`** 去 **`--cpu`**），**新开登记行**；**非**主线阻塞；**`RETRIEVAL_HEAD_NOTES.md`**、**`NEXT_EXPERIMENTS_COMMANDS.md` §6**。  
3. **A2-S3 可选加压**：更大 **`heldout-leaves`**、**`root_child`**、**stratified + `split-seed`** — 与 **init×5** **分列** 说明。  
4. **Polish**：**S5 总表**（**`RESEARCH_NOTES` §7**）、主图入仓、平面 RAG smoke。  
5. **SSGS（辅线，非阻塞）**：**Wikitext 同树** 已归档 **`ssgs_mamba_wikitext_grid.csv`**（**n8–64** 等，登记 **X-20260407-ssgs-mamba-wikitext-tree**）。可选：**n=128**、**`git pull` 后** 重跑 **一格** 刷新 **`git_sha`**；玩具树 **X-20260421**、LM 并列 **X-20260425** 仍足 **附录** 基线。  
6. **总览**：**`RESEARCH_STATUS_AND_DIRECTION.md`**、**`NEXT_RESEARCH_PLAN.md`**（**篇首「当前收口清单」**）；手册：**`SERVER_SWEEP_RUNBOOK.md`**、**`NEXT_EXPERIMENTS_COMMANDS.md`**（**§11 本机 5060**）、**`LOCAL_5060_RUNBOOK.md`**、**`RUN_AUTOADL_SECTION7_NOW.md`**；草稿：**`PHASE2_DRAFT.md`**、**`FIGURE_CAPTIONS_STAGE1.md`**。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初稿：与 `CURRENT_SPRINT` 主线收束、`metrics_result` 归档同步 |
| 2026-04-09 | 增加 §8 指向 **`NEXT_RESEARCH_PLAN.md`** |
| 2026-04-07 | §5 增 **5060 CUDA** Wikitext **n8/c12** 动机 JSON；§8 增 **`PHASE2_DRAFT.md`** / A2-S3 指针 |
| 2026-04-07 | §5 增 **5060 CUDA** Wikitext **n16/c12** 动机 JSON |
| 2026-04-07 | §5 增 **5060 CUDA** Wikitext **n16/c8** 动机 JSON |
| 2026-04-07 | §5：**5060 CUDA** Wikitext **2×2** 四 JSON + 指针 **`PHASE2_DRAFT` §1.1** |
| 2026-04-07 | §3.1 **5060** 四格动机段；§5 增 **grid CSV**；**`path_pair_geometry`** + 单测；**A2-S3** `chunk_len=12` leaf_heldout |
| 2026-04-07 | **P1 成文**：新增 **§8** 阶段 2（系统+A2-S3）、**§9** 检索头/Mamba 讨论边界；**§10** 指针；摘要/英文摘要各增一句；原 **§8** 下移为 **§10** |
| 2026-04-07 | **§8.3**：**A2-S2** 待云端段落改为 **`run_server_stage2_wikitext_grid.sh`** + **`SERVER_SWEEP_RUNBOOK` §2d** |
| 2026-04-09 | **§8.3**：**A2-S2** 实测归档（**`STAMP=20260409T1035Z`**）+ **`EXPERIMENT_REGISTRY` A-stage2-wikitext-grid-v1** 更新 |
| 2026-04-09 | **§8.3**：**A2-S2 R2**（**`STAMP=20260409T1110Z`** **`stage2_fused_r2`**）；峰值与 R1 一致；登记册合并叙述 |
| 2026-04-09 | **§8.3**：**dim256 四格**、**32 叶单点**、**headcheck**；新登记 **A-stage2-wikitext-dim256-v1**、**n32-c8**、**X-20260409-wikitext-headcheck** |
| 2026-04-09 | **成文收口**：摘要/§3/§4/§5/**§8.3 压缩**/**§8.4 新增**/**§10 下一步**；叶数扫描、XL、§7 **depth 5–6** 入表；**§10** 增 **A2-S3 → `NEXT_EXPERIMENTS_COMMANDS` §9** |
| 2026-04-09 | **§8.2 / §10**：**leaf_heldout** 多种子 = **`--init-seed`**；**`split-seed`** 仅 **`stratified`** |
| 2026-04-09 | **§5 / §10**：**A2-S3** **3090 init×5** 入 **`EXPERIMENT_REGISTRY`**；**`aggregate_task_wikitext_path_pair_json.py`** |
| 2026-04-09 | **§10**：**`STAMP=20260409T1438Z`** 归档 **`metrics_result/`**；**主线成文优先**；**B-S2+** 标为 **可选机制线** |
| 2026-04-09 | **精修**：摘要/结论/英摘；**§4** 增 **SSGS** 定位与 **是否续做**；**§8.2** 增 **1438Z** 聚合一句；**§10** 增 **SSGS 可选续作** |
| 2026-04-07 | **§10 / §4 SSGS**：**Wikitext 同树** — **`demo_ssgs_mamba_wikitext.py`**、登记 **X-20260407-ssgs-mamba-wikitext-tree** |
| 2026-04-07 | **成文收口**：摘要/§4/§5/§6/§7 英摘/§10 补 **Wikitext SSGS grid**；**§5** 表增 **`ssgs_mamba_wikitext_*`** |
| 2026-04-07 | **§10**：**`NEXT_RESEARCH_PLAN.md`** 增 **「当前收口清单」**（成文 / 仓库 hygiene / 可选 GPU 一条） |
| 2026-04-07 | **§5.1 成文核对**：主图 PNG、CSV、§7、5060、SSGS、A2-S3 **路径核对表**；**脚注矩阵**一句 |
| 2026-04-07 | **§5 表**：**dim128** 主文 CSV 实际文件名为 **`paper_main_dim128_localgrid_*`**（与 dim256/384 **分列一行** 说明） |
| 2026-04-07 | **§10**：**`LOCAL_5060_RUNBOOK.md`** + **`NEXT_EXPERIMENTS` §11**（本机 5060） |
| 2026-04-07 | **§5 / §5.1**：**B-S2+** **`…local5060.json`**；登记 **X-20260407-local5060-bs2plus-rerun** |
| 2026-04-07 | **§9.1**：**5060 CPU B-S2+** 成文 + **§7** 英摘一句；**§10** 第 2 条更新（CPU 已归档） |
