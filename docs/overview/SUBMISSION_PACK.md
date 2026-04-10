# 投稿成文包（P0：A1–A4）

> **用途**：把 **`PHASE1_MANUSCRIPT.md`**、**`FIGURE_CAPTIONS_STAGE1.md`**、**`EXPERIMENT_REGISTRY.md`** 压成 **可粘贴** 的叙事与脚注；**登记真相**仍以 **`EXPERIMENT_REGISTRY.md`** 为准。  
> **生成**：按 **`NEXT_RESEARCH_PLAN.md`** **「无云端时：标准推进」** §A 维护；路径核对于 **2026-04-10** 对仓库做一次存在性检查。

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

---

## A2 · 归档路径核对（存在性自检）

| 类别 | 路径（相对仓库根） | 状态 |
|------|---------------------|------|
| 主图 PNG ×3 | `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png` | 已存在 |
| 主文 CSV fused | `results/metrics_result/paper_main_dim128_localgrid_paper_main_v1.csv`；`paper_main_dim{256,384}_paper_main_v1.csv` | dim128 已抽检；余请投稿前再扫 |
| 主文 CSV naive | `paper_main_*_paper_main_naive_v1.csv`（同上 dim 模式） | 投稿前全量核对 |
| Manifest | `results/metrics_result/paper_main_manifest_paper_main_{v1,naive_v1}.txt` | 已存在 |
| 阶段 2 叶数扫描（例） | `results/metrics_result/benchmark_wikitext_stage2_leavescale_20260409T1257Z_n{8,16,32,64}_c8.json` | n8–64 四文件已存在 |
| SSGS grid | `results/metrics_result/ssgs_mamba_wikitext_grid.csv` | 投稿前核对登记行 |
| 本机 5060 登记 JSON | `results/metrics/probe_path_reader_linear_text*_heldout_local5060.json`、`task_wikitext_sibling8_local5060_cpu_20260410.json` 等 | 见 **EXPERIMENT_REGISTRY** **X-20260410-*** |

**动作**：投稿前用 **`git status`** + **`PHASE1_MANUSCRIPT` §5.1** 全表再扫一遍；重跑数据后以 JSON 内 **`git_sha`** 为准更新正文。

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
