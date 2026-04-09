# 阶段 1 成文稿（可直接迁入论文/技术报告）

> **用途**：将阶段 1 **系统验证**整理为连续叙述；数字与路径以 **`EXPERIMENT_REGISTRY.md`** 与本文 **§5 归档索引** 为准。  
> **勿与 §7 玩具协议混读**：path-batch 主结果与 S1–S4 **各列毫秒** 的物理含义不同，见 **`FIGURE_CAPTIONS_STAGE1.md`** 篇首与 **`RESEARCH_NOTES.md` §7.0**。

---

## 摘要（约 150 字）

在树状索引上，以 **同一批根—叶路径** 对 **Transformer、GRU、Mamba-2** 三类路径 reader 做批量前向，测量 **步均耗时** 与 **CUDA 峰值显存**。实验表明：**Mamba-2 path reader 的可观测效率对是否启用 `mamba_ssm` 融合实现极为敏感**——无融合核时峰值可达 **GiB** 量级，与 **Transformer/GRU** 的 **百 MiB** 量级形成反差；融合后同网格峰值可降至 **约 51–217 MiB**。故在树形 RAG 的 path-batch 负载下，**不能**仅用 SSM 名义复杂度替代实测效率；主文须 **同机、同 commit** 报告 naive 与 fused 对照。

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
3. **语料泛化（浅树）**：在 **Wikitext-2 叶块** 构造的浅树上使用 **同一 reader 槽位**（登记 **A-20260408-wikitext-3090-fused**），支持「不止合成树」的叙述。

---

## 4 与 §7 玩具协议的关系（一段话）

**§7**（S1–S4）在 **单条** 合成路径上分别测量 **Mamba `DynamicCache` clone**、**Transformer 整段重算（TF-R1）**、**带 KV 的增量前向（TF-KV）**、**快照 restore** 等，**各列不可互换**。该协议用于 **机制层与回溯叙事**，**补充** path-batch 主图，**不替代**主图曲线。已在 **CUDA** 上 **串行复跑**（`run_path_protocol_cuda.sh`），新 JSON 与 **`RESEARCH_NOTES` §7.3.1** 及仓内 `*_20260421.json` **同阶**，详见 **`PHASE1_COMPLETE_SUMMARY.md` 附录 B**。

---

## 5 数据归档索引（本仓库）

**统一目录**：`results/metrics_result/`（本机示例：`D:\cursor_try\mamba2\results\metrics_result`）。**登记与正文引用时请写相对仓库根的路径。**

| 类别 | 文件名模式 | 说明 |
|------|------------|------|
| **主文 fused CSV** | `paper_main_dim{128,256,384}_paper_main_v1.csv` | 3090 fused 网格 |
| **主文 naive CSV** | `paper_main_*_paper_main_naive_v1.csv` | 同机 naive |
| **Manifest** | `paper_main_manifest_paper_main_{v1,naive_v1}.txt` | 环境元数据 |
| **主图** | `../metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png` | 与 **pair** 登记对应 |
| **§7 复跑（2026-04-08）** | `mamba2_cache_snap_segments_depth4_cuda_20260408T1617Z.json` 等 6 文件 | S1–S4 + branchdemo；另含 `*_20260408T1030Z` 为同协议早次复跑 |
| **Wikitext** | `benchmark_wikitext_3090_fused_20260408T0846Z.json` | 浅树 fused |
| **Wikitext（5060 CUDA，动机）** | `benchmark_wikitext_5060_cuda_n8_c12_20260407.json` | **HF naive** Mamba；**不可**与上行 3090 fused **同表混点** |
| **大叶数研究** | `sweep_research_large_leaves_*_research_lg_v1.csv` + manifest | 登记 **A-20260408-research-large-leaves-3090** |

**历史归档**（仍在 `results/metrics/`）：`**_20260421.json** 系列，与 **X-20260421-*** 登记一一对应；与 `metrics_result` 中 **STAMP** 文件 **并存**，便于 diff。

---

## 6 结论段（约 200 字，与 `PHASE1_VALIDATION_PLAN.md` §6.3 一致）

在树路径批量编码设定下（`fanout=2`、`dim=128`、多组 `depth×chunk_len`），HuggingFace **无融合核** 的 Mamba-2 path reader 的 CUDA **峰值显存** 可达 **GiB 级**（例如 64 条并行路径时约 **8.9GiB**），而同设定下 **Transformer / GRU** 路径 reader 多在 **百 MiB** 量级。相对地，在 **AutoDL** 上启用 **`mamba_ssm` 融合实现** 后，**同一网格**上 Mamba2 峰值可降至 **约 51–217MiB**（随配置变化），降幅可达 **两个数量级**。因此：**在树形 RAG 的 path-batch workload 中，可观测效率高度依赖实现是否融合，而不能仅由 SSM 架构名义复杂度替代。**

---

## 7 英文摘要（可选，约 120 词）

We benchmark Transformer, GRU, and Mamba-2 **path readers** on tree-structured retrieval paths under a fixed **path-batch** harness, reporting step time and **CUDA peak memory** (`max_memory_allocated`). On identical grids, **HuggingFace Mamba-2 without fused kernels** can reach **GiB-scale** peaks whereas **Transformer/GRU** stay around **hundreds of MiB**; with **`mamba_ssm` + `causal_conv1d`**, Mamba-2 peaks drop to **tens to low hundreds of MiB**—often **two orders of magnitude** lower than naive. Thus, **observed efficiency is implementation-dependent** and must not be inferred from asymptotic claims alone. A separate **toy protocol** (§7) measures clone/restore/KV-increment **per column**; it **does not** merge with path-batch figures.

---

## 8. 下一阶段（指针）

1. **总览（先读）**：**`docs/overview/RESEARCH_STATUS_AND_DIRECTION.md`** — 现状、整体方向、决策规则、**推荐执行顺序**（含 **五条测量轴**）。  
2. **任务展开**：**`docs/overview/NEXT_RESEARCH_PLAN.md`** — 里程碑 A2-S0…、B-S1…、命令与风险。  
3. **阶段 2 叙事与任务（效果 proxy）**：**`docs/overview/PHASE2_DRAFT.md`** — 真语料树 + **A2-S3**（**`scripts/research/task_wikitext_path_pair.py`**；归档 **`results/metrics/task_wikitext_*.json`**）；与本文 path-batch **分列**，见草稿 **§2–§3**。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初稿：与 `CURRENT_SPRINT` 主线收束、`metrics_result` 归档同步 |
| 2026-04-09 | 增加 §8 指向 **`NEXT_RESEARCH_PLAN.md`** |
| 2026-04-07 | §5 增 **5060 CUDA** Wikitext **n8/c12** 动机 JSON；§8 增 **`PHASE2_DRAFT.md`** / A2-S3 指针 |
