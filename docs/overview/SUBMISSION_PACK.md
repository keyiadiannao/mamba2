# 投稿成文包（P0：A1–A4）

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

## 修订

| 日期 | 说明 |
|------|------|
| 2026-04-10 | 初版：A1–A4；对接 **`NEXT_RESEARCH_PLAN`** **无云端 §A** |
| 2026-04-10 | **A1b** 中英摘要/引言提示；**A2** 全表 **✅** 存在性扫描（paper_main 6 CSV、5060 四 JSON+grid、leavescale n8–64、SSGS grid） |
