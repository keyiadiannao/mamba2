# 投稿成文包（P0：A1–A7）

> **用途**：把 **`PHASE1_MANUSCRIPT.md`**、**`FIGURE_CAPTIONS_STAGE1.md`**、**`EXPERIMENT_REGISTRY.md`** 压成 **可粘贴** 的叙事与脚注；**登记真相**仍以 **`EXPERIMENT_REGISTRY.md`** 为准。  
> **生成**：按 **`NEXT_RESEARCH_PLAN.md`** **「无云端时：标准推进」** §A 维护；路径核对 **已扫仓库**（**2026-04-10**）。

---

## A1 · 一页故事线（问题 → 方法 → 结果 → 边界）

**问题**  
在**树状索引**上，对一批**根—叶路径**做 **path-batch 编码**时，**Mamba-2** 与 **Transformer / GRU** 路径 reader 的 **步均耗时**与 **CUDA 峰值显存**如何变化？该问题**不**等价于「全模型 KV 总账」或「端到端 RAG 吞吐」。

**方法**  
固定 **建树与遍历协议**，使用 **`run_tree_reader_benchmark` / `benchmark_wikitext_tree.py`** 同一槽位，报告 **`per_step_s`、`peak_alloc_mib`（`max_memory_allocated`）**；**3090** 上 **同网格、同 commit** 对照 **HF naive Mamba** 与 **`mamba_ssm` fused**。阶段 2 在 **Wikitext-2 浅树**上扩展 **网格与叶数**；**A2-S3** 为 **叶对 cohort + 岭回归** 的 **效果 proxy**（与墙钟 **分列**）。**§7**、**SSGS** 为 **机制/导航辅线**，**非**主图同一纵轴。

**主要结果**  
- **实现敏感性**：naive 与 fused 下 Mamba2 **峰值**可差 **约两个数量级**；**不得**仅用 SSM 名义复杂度代替实测。  
- **真语料**：Wikitext **同 harness** 下已归档 **四格、dim256、叶数 8→256、§7 depth 5–6** 等（见 **§A2 核对表**）。  
- **A2-S3（示例）**：**leaf_heldout**、多 **init_seed** 下 **GRU** 等小样本 test 量级见 **`PHASE1_MANUSCRIPT` §8.2**；**本机 n8 stratified** 与 **3090** 设置 **分列**，不得混为同一「难度」。

**边界（须在摘要或脚注出现）**  
- **5060 + HF naive** 与 **3090 + fused** **禁止无脚注同表**。  
- **path-batch**、**§7 毫秒列**、**SSGS 快照/回滚计数**、**A2-S3 准确率** 为 **四条（及以上）独立测量轴**。

### A1b · 摘要 / 引言用草稿（摘自 `PHASE1_MANUSCRIPT.md`，可再压缩到字数限制）

**中文摘要（约 150 字量级；投稿前按会议删改）**

在树状索引上，以同一批根—叶路径对 Transformer、GRU、Mamba-2 三类路径 reader 做批量前向，测量步均耗时与 CUDA 峰值显存。实验表明：Mamba-2 path reader 的可观测效率对是否启用 `mamba_ssm` 融合实现极为敏感——无融合核时峰值可达 GiB 量级，与 Transformer/GRU 的百 MiB 量级形成反差；融合后同网格峰值可降至约 51–217 MiB（随叶数与 dim 可升至数百 MiB–约 1 GiB 以下，仍与 naive 分列）。故在树形 RAG 的 path-batch 负载下，不能仅用 SSM 名义复杂度替代实测效率；主文须同机、同 commit 报告 naive 与 fused 对照。阶段 2 在 Wikitext-2 浅树上沿用同一 harness，已归档四格、dim256、叶数 8→256 等登记级效率曲线；并以叶对 cohort + 岭回归给出与墙钟分列的效果 proxy。§7 玩具协议已扩展 depth 5–6；SSGS×Mamba 与主图非同一 harness；Wikitext 同树上已归档 snapshots/rollbacks 与 grid（§4、§5）。

**英文摘要（约 120 词；见 `PHASE1_MANUSCRIPT.md` §7 英文摘要全文）**

可直接复制 **`PHASE1_MANUSCRIPT.md`** 从 *We benchmark Transformer, GRU…* 起一整段；投稿前统一 **时态** 与 **期刊缩写**。

**引言首段提示（非正文）**  
先落 **树路径编码 + path-batch harness**，再点出 **naive vs fused** 与 **真语料扩展**；**勿**在首段混谈 Agent 端到端或「检索头」——细节放 **§9 / 附录**。

---

## A2 · 归档路径核对（存在性自检）

**仓库扫描（2026-04-10）**：下列路径均已 **存在**（相对仓库根）。

| 类别 | 路径 | 状态 |
|------|------|------|
| 主图 PNG ×3 | `results/metrics/figures/mamba_3090_naive_vs_fused_dim128_paper_main_v1.png` 等 dim256/384 | ✅ |
| 主文 CSV **fused** | `paper_main_dim128_localgrid_paper_main_v1.csv`、`paper_main_dim256_paper_main_v1.csv`、`paper_main_dim384_paper_main_v1.csv` | ✅ |
| 主文 CSV **naive** | `paper_main_dim128_localgrid_paper_main_naive_v1.csv`、`paper_main_dim256_paper_main_naive_v1.csv`、`paper_main_dim384_paper_main_naive_v1.csv` | ✅ |
| Manifest | `paper_main_manifest_paper_main_v1.txt`、`paper_main_manifest_paper_main_naive_v1.txt` | ✅ |
| 5060 Wikitext 四格 JSON | `benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json` | ✅ |
| 5060 汇总 CSV | `benchmark_wikitext_5060_cuda_grid_20260407.csv` | ✅ |
| 阶段 2 叶数扫描 n8–64 | `benchmark_wikitext_stage2_leavescale_20260409T1257Z_n{8,16,32,64}_c8.json` | ✅ |
| SSGS grid | `ssgs_mamba_wikitext_grid.csv` | ✅ |

**投稿前仍须人工核对**：正文引用的 **每一个** 文件名与 **`PHASE1_MANUSCRIPT` §5.1** 表 **逐字一致**；若重跑数据，以 JSON **`git_sha`** 更新 **方法/附录**。

**本机 5060 登记 JSON**（成文脚注用）：**`EXPERIMENT_REGISTRY`** **X-20260410-***、**X-20260407-local5060-bs2plus-rerun**；路径见 **`LOCAL_5060_RUNBOOK.md`**。

**动作**：投稿前 **`git status`** 干净；**§7.5 S5** 总表若补，另开一行登记（**可选**）。

---

## A3 · 可粘贴：叙事边界与五轴（中文）

**一段边界（摘自 `FIGURE_CAPTIONS_STAGE1.md` P0，可放 Related Work 或 Method 脚注）**

主文主图呈现的是 **path-batch** 下三 reader 的 **时间与 Mamba2 峰值显存**；**不实现**树上 DFS 试错序，也**不把**全模型 KV 分项摊进同一张主图。**§7 玩具协议** 在 **单条路径**上 **分列**测量 clone / restore / TF-R1 / TF-KV 等，**各列物理含义不同**。**SSGS** 线报告 **DFS + token 步进**下的 **快照/回滚计数**，**不是** path-batch 墙钟。**阶段 2 任务（A2-S3）** 为 **岭回归准确率类 proxy**，**不是**主图纵轴。

**五轴一句（防混读）**  
正文 **禁止**将 **path-batch 主图**、**§7 玩具表各列**、**SSGS 计数**、**真 LM 玩具线**、**A2-S3 准确率** 的纵轴或列 **无标注合并** 或 **相减**（详见 **`FIGURE_CAPTIONS_STAGE1.md`** 表）。

**5060 与 3090**  
**5060** 上 **HF naive** 的 Wikitext 动机数字 **仅**作 **本地动机/趋势**；**3090 fused** 主文格点 **须分列脚注**，**禁止**无标注混点（**`PHASE1_MANUSCRIPT` §5.1**）。

---

## A4 · 附录可用：检索头文献 vs 本文探针（短段）

**可放附录「与 Retrieval Head 的关系」**（压缩自 **`PHASE1_MANUSCRIPT` §9**）

文献中的 **Retrieval Head** 多在 **Decoder 多头自注意力**上通过 **注意力权重** 定义「头」的角色。**本文** path reader 中的 **Mamba-2** **不提供**与之 **同构** 的 **逐头 QK 图**；**`num_heads`/`head_dim`** 为 **SSD/Mamba-2 块内**划分，**不可**直接等同于上述「检索头」。**B-S2 / B-S2+** 探针（**`probe_retrieval_correlation` / `probe_path_reader_linear`**）回答的是 **池化表示上合成标签是否线性可读**，属 **表征级** 证据，**不是** **头级因果** 结论。表述宜用 **「层状探针 / 线性可读性」**，**避免** **「Mamba 的检索头」** 除非另给 **操作定义**。

**文献细节表**：**`docs/research/RETRIEVAL_HEAD_NOTES.md`**（尤其 **§8**）。

---

## 提交前检查（代码与健康；不占云端）

| 检查 | 命令 / 动作 |
|------|-------------|
| **单元与协议测试** | **`mamba2` Python**：`python -m pytest tests/ -q`（**20** passed 量级） |
| **无 torch 快测** | `py -3 -m pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py -q` |
| **工作区** | **`git status`** 干净；勿手改 **`metrics_result/`** 归档 |
| **可选本机复跑** | **`LOCAL_5060_RUNBOOK.md` §5.1**（§7 S1/S2 CPU、B-S2 gpt2 topic） |

---

## A5 · 正文骨架（中文；按节粘贴后改术语与时态）

**1 引言**  
树形索引下，检索常归结为沿 **根—叶路径** 编码文本；**path-batch** 将多条路径并行前向。**Mamba-2** 以递归状态更新替代全长 KV，理论上利于长路径，但 **工程实现（是否融合）** 对 **显存与延迟** 的影响需在固定 harness 下 **实测**。本文在 **与 Transformer / GRU 同槽位** 的 reader 上，报告 **3090 同机 naive vs fused** 及 **Wikitext-2 浅树** 扩展；并给出 **与墙钟分列** 的 **A2-S3** 任务 proxy 与 **§7 / SSGS** 辅线（**测量轴分列**，见 **§A3**）。

**2 方法（系统）**  
- **树与路径**：平衡 `k` 叉树；节点为 `chunk_len×dim` 嵌入（合成或 Wikitext 叶块确定性嵌入）。  
- **Path-batch 基准**：对给定路径集合调用 **`run_tree_reader_benchmark`** / **`benchmark_wikitext_tree.py`**，记录 **`elapsed_s`、`per_step_s`、CUDA 上 `max_memory_allocated` 峰值**。  
- **naive vs fused**：**同一 GPU、同一网格、同一 `WARMUP`/`REPS`**；naive 为 HF 回退，fused 为 `mamba_ssm`+`causal_conv1d`（以 **manifest** 为准）。  
- **阶段 2**：Wikitext **同 harness** 扩展 **叶数与 dim**；**A2-S3** 为 **叶对 cohort + 岭回归**（**非** path-batch 纵轴）。

**3 结果（主文，占位）**  
- **Fig.1**（三张 PNG）：**dim128/256/384** 上 Mamba2 **峰值** naive vs fused；**结论句**见 **`PHASE1_MANUSCRIPT` §6**。  
- **表 / 段**：Wikitext **fused** 曲线或代表格点（**`metrics_result/benchmark_wikitext_stage2_*`**）；**5060 naive** 动机 **单独脚注**，**不与 3090 fused 混表**。  
- **A2-S3**：**3090 heldout + 多种子** 与 **本机 n8 stratified** **分列**叙述（**`PHASE1_MANUSCRIPT` §8.2**）。

**4 讨论与局限**  
- **实现依赖性**：主结论为 **可观测效率依赖融合实现**，非「Mamba 对 Transformer 绝对优劣」。  
- **规模**：小 encoder、浅 reader；**不**直接外推 **7B+ 端到端 RAG**（见 **`RESEARCH_STATUS_AND_DIRECTION.md` §6**）。  
- **检索头表述**：见 **§A4**；**B-S2/B-S2+** 为 **表征级** 探针。  
- **§7 / SSGS**：机制与导航 **过程量**；**禁止**与主图毫秒/峰值 **无标注混读**。

---

## A6 · 结果陈述模板（可直接改成论文章节句）

- **R1（主）**：在 **path-batch** 设定与 **我们的网格** 上，**HF naive Mamba-2** 的 **CUDA 峰值** 可达 **GiB** 量级，而 **同设定下 Transformer/GRU** 多在 **百 MiB**；**启用 `mamba_ssm` 融合** 后，**同机同网格** Mamba2 峰值降至 **约 10¹–10² MiB** 量级（随 **叶数、dim** 变化），与 naive **分列**报告。  
- **R2（真语料）**：**Wikitext-2 浅树** 上 **同 harness** 的 **叶数扫描与 XL** 表明 **fused** 下峰值仍随规模上升；**步均耗时** 不作「Mamba 全面最快」的绝对宣称（见 **`PHASE1_MANUSCRIPT` §8.4** 写作提示）。  
- **R3（任务 proxy）**：**A2-S3** 在 **小样本 held-out** 上给出 **ridge test_acc** 量级；**任务 v0、未训练 reader**，**不与 R1 混表**。  
- **R4（辅线）**：**§7** 给出 **单路径**上 **clone/restore/TF-R1/TF-KV** 等 **分列**测量；**SSGS** 给出 **快照/回滚计数**（**`ssgs_mamba_wikitext_grid.csv`**）。

---

## A7 · 本机实验：高性价比 vs 搁置（无云端）

| 策略 | 内容 |
|------|------|
| **值得做** | **§7 与 S1 对称的微基准**（如 **S2 TF-R1 CPU**）、**B-S2 一次复跑**、**`pytest tests/`** —— **成本低**，且 **补附录/脚注** 与 **`RESEARCH_NOTES` §7** 对齐。 |
| **搁置（性价比低）** | **重复** path-batch / A2-S3 / B-S2+ **已有 JSON**；**CPU 上大规模 Wikitext 全网格**；**TF-KV / S4 restore CPU 全量**（与 **3090 已归档** 重复、本机耗时高）；**SSGS n128 CPU**；**topic `sample` 与 `heldout` 双份** 若无审稿需求。 |

**云端恢复后优先**：**P1** B-S2+ CUDA 一行；**P2** SSGS sha / n128（**CUDA**）。

---

## 修订

| 日期 | 说明 |
|------|------|
| 2026-04-10 | 初版：A1–A4；对接 **`NEXT_RESEARCH_PLAN`** **无云端 §A** |
| 2026-04-10 | **A1b** 中英摘要/引言提示；**A2** 全表 **✅** 存在性扫描（paper_main 6 CSV、5060 四 JSON+grid、leavescale n8–64、SSGS grid） |
| 2026-04-10 | **提交前检查**表（pytest / git）；与 **`LOCAL_5060_RUNBOOK` §5.1** 对齐 |
| 2026-04-10 | **A5–A7**：正文骨架（引言/方法/结果/讨论）、**A6** 结果句模板、**A7** 本机 **高性价比/搁置** |
