# 主线叙事链：树 → 路径 → reader → 导航（+ SSGS）

> **用途**：把投稿主线的 **概念顺序** 与 **本仓证据/数据** 一一对应，避免与「全模型 KV 总账」「端到端 Agent」混读。  
> **权威对齐**：**`PROJECT_MASTER_PLAN.md` §1.0**（主验证轴）；**数字与路径真相** → **`EXPERIMENT_REGISTRY.md`**、**`DATA_ARCHIVE_202604_SERVER.md`**；**七轴防混** → **`FIGURE_CAPTIONS_STAGE1.md`**。

---

## 0. 摘要可用稿（可直接迁入摘要 / Abstract）

**用法**：下列段落与 **`FOUNDATION_STAGE_FORMAL_SUMMARY.md`**、**`PHASE1_MANUSCRIPT.md` 摘要** 同轴；**具体数字、峰值区间、登记 STAMP** 仍以 **`EXPERIMENT_REGISTRY`** 与 **`metrics_result`** 为准；**各测量轴须脚注分列**，勿把 SSGS/M1 与 path-batch **混为同一纵轴**。

### 0.1 中文摘要（约 220–280 字）

在**树状索引**上，检索常归结为沿**根—叶路径**编码文本块。本文在固定 **path-batch** 协议下，对 **Transformer、GRU、Mamba-2** 三类 **path reader**（同槽位）批量前向，报告步均耗时与 **CUDA 峰值显存**。结果表明：**Mamba-2 的可观测效率对是否启用融合实现极为敏感**——无融合核时峰值可达 **GiB** 量级，与 Transformer/GRU 的 **百 MiB** 量级形成反差；融合后同网格峰值可降至 **约数十至数百 MiB**（仍随叶数与维度变化），故**不能**以 SSM 名义复杂度替代**实测**效率；主文须**同机、naive/fused 分列**。在 **Wikitext-2 浅树**上扩展同 harness 效率曲线，并辅以**与墙钟分列**的任务 proxy 与机制测量。**SSGS** 在 **DFS 试错序**下给出 **快照/回滚计数**；**Phase M1** 在**同树、同 DFS 任务**上对照 **SSGS 与玩具 TF-KV**（clone / truncate_kv），讨论路径级**回溯与 KV 类基线的代价形状**。上述辅线与主效率图、§7 玩具协议**各为独立轴**。**局限**：小宽度编码器；结论**不**自动推广至全规模 LLM KV 与端到端 Agent。

**关键词**：树状检索；路径批量编码；Mamba-2；实现敏感性；状态快照；SSGS；受控对照

### 0.2 Abstract（English, ~160–200 words)

Tree-structured retrieval often reduces to encoding text along **root-to-leaf paths**. We benchmark **Transformer, GRU, and Mamba-2 path readers** under a fixed **path-batch** harness, reporting per-step time and **CUDA peak memory** (`max_memory_allocated`). **Mamba-2 peak memory is highly sensitive to implementation**: **HuggingFace-style runs without fused kernels** can reach **GiB-scale** peaks, whereas **Transformer/GRU** stay near **hundreds of MiB** on the same grids; with **`mamba_ssm` fused kernels**, Mamba-2 peaks typically fall to **tens–hundreds of MiB** (still **configuration-dependent**). Thus **observed cost cannot be inferred from nominal SSM complexity alone**; we report **naive vs fused on the same machine** as separate rows. On **Wikitext-2 shallow trees**, we extend the **same harness** with registry-scale grids and **task / mechanism lines reported on separate axes**. **SSGS** reports **DFS-ordered** navigation with **snapshot/rollback counts**; **Phase M1** compares **SSGS Mamba** with a **toy TF-KV** baseline (**clone / truncate_kv**) on **one tree and one DFS goal**, characterizing **backtracking vs KV-style costs** at the **path / toy-trunk** level. **Auxiliary lines are not merged** with the main wall-clock figures without footnotes. **Limitations**: **small encoders**; claims **do not** automatically extend to **full-model KV accounting** or **end-to-end Agent** systems.

**Keywords**: tree-structured retrieval; path-batch encoding; Mamba-2; implementation sensitivity; state snapshots; SSGS; controlled comparison

### 0.3 成文提示（与主线推进挂钩）

- **引言首段**可用 **§1 叙事链** 四句拆写；**摘要**宜保留 **path-batch 主结论 + naive/fused + 一句辅线（SSGS/M1）+ 局限**。  
- **勿在摘要内**写具体 MiB/GiB 格点（除非 venue 要求）；**数值**放正文与 **`paper_main_*` CSV**。

---

## 1. 叙事链（一段话）

**树状 RAG**：语料被组织成 **树索引**；检索或导航往往归结为沿 **根—叶路径** 依次读 **chunk** 嵌入。  
**路径上的 reader**：在固定 harness 里，用 **Mamba-2 / Transformer / GRU** 三类 **path reader**（**同一槽位**）对一批路径做批量前向，度量 **延迟与 CUDA 峰值显存**；**Mamba** 须 **naive 与 fused 分列**，不能把结论写成单靠「SSM 名义复杂度」。  
**导航（DFS 试错序）**：在树上按 **深度优先** 试探时，走错子树需要 **回溯**。**SSGS** 用 **Mamba `DynamicCache` 的快照 / 恢复** 实现「悔棋」，并记录 **snapshots / rollbacks / leaf_checks** 等 **迹**（**不是** path-batch 墙钟）。  
**论文要讲的代价结构**：**树**提供搜索结构；**Mamba**提供 **固定大小隐状态** 的递推更新；**SSGS**利用该状态做 **相对可控的** 回溯代价叙事——与 **「整段 KV 随上下文线性增长」的玩具对照（M1）** 并列时，讨论的是 **路径级 / 玩具 trunk** 上的 **代价形状**，**不是** 全规模 LLM 的 KV 分项总账，也 **不是** 端到端 RAG 吞吐的单一数字。

---

## 2. 概念层 → 本仓落点（四条链，脚注分列）

| 层 | 叙事角色 | 本仓做什么 | **不是**什么 |
|----|-----------|------------|----------------|
| **树** | 索引结构 | **合成平衡树**、**Wikitext-2 叶块 → 自底向上建树**（与 `benchmark_wikitext_tree` 同协议处对齐） | 不等同于工业 **RAPTOR 全流水线** 的唯一实现 |
| **路径** | 根—叶 chunk 序列 | **path-batch**：对给定路径集合 **批量前向**（**`run_tree_reader_benchmark` / `benchmark_wikitext_tree`**） | **不实现** 树上 **DFS 试错序** 本身 |
| **reader** | 序列编码槽位 | **Mamba2PathReader** vs **Transformer / GRU**；**naive vs fused** 同机对照 | 不是 7B **端到端** 的 **唯一骨干** 对打（见 **`RESEARCH_STATUS` §6**） |
| **导航 + 回溯** | DFS + 悔棋 | **SSGS**：**`dfs_ssgs_mamba`** + Wikitext 同树（**`ssgs_mamba_wikitext_grid.csv`**）；**M1**：同树 **DFS 任务** 上 **SSGS vs 玩具 TF-KV**（**`ssgs_vs_kv_wikitext_nav_grid.csv`**） | **不是** path-batch 主图纵轴；**玩具 TF-KV ≠** 全模型 KV |

**副线**（不替代上述主链第一叙事）：**检索头 / B-S2+ 探针**、**A2-S3 岭回归 proxy**、**§7 单列毫秒** — 各为 **独立测量轴**，见 **`FIGURE_CAPTIONS_STAGE1.md` 七轴表**。

---

## 3. 数据与登记索引（可粘贴进方法/附录脚注）

**原则**：下列 **basename** 以仓内 **`results/metrics_result/`** 为准；**行数 N** = 仓根 **`aggregate_*` stdout 的 `N row(s)`**（勿写死旧数字）。

### 3.1 Path-batch（效率主图）

| 内容 | 登记 / 索引 | 代表路径 |
|------|----------------|----------|
| 3090 naive vs fused 主曲线 | **`A-20260408-paper-main-3090-*`** 等 | **`paper_main_*_{v1,naive_v1}.csv`**；**`results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`** |
| Wikitext 同 harness 扩展 | **A-stage2-***、**`DATA_ARCHIVE` §2.3** | **`benchmark_wikitext_stage2_*`**、**leavescale / XL** JSON + CSV |

### 3.2 SSGS × Mamba × Wikitext（迹，非墙钟）

| 内容 | 登记 id | 汇总表 / 通配 |
|------|---------|----------------|
| DFS + 快照/回滚计数 | **`X-20260407-ssgs-mamba-wikitext-tree`** | **`ssgs_mamba_wikitext_*.json`** → **`ssgs_mamba_wikitext_grid.csv`**（**`aggregate_ssgs_mamba_wikitext_json.py`**） |
| 同树 path-batch smoke（辅） | 见登记册 / **`SSGS_MAINLINE_M1.md`** | **`benchmark_wikitext_ssgs_bundle_*`**（与 SSGS **分列**） |

### 3.3 Phase M1（同树 DFS：SSGS vs 玩具 TF-KV）

| 内容 | 登记 id | 汇总表 / 通配 |
|------|---------|----------------|
| 三臂 DFS + 可选 L3 | **`X-ssgs-vs-kv-tree-nav-m1`** | **`ssgs_vs_kv_tree_nav_wikitext_*.json`** → **`ssgs_vs_kv_wikitext_nav_grid.csv`**（**`aggregate_ssgs_vs_kv_wikitext_json.py`**） |
| **`chunk_len=12` 消融**（与默认 c8 **分列**） | 同上（脚注 **M2 §Ⅲ-1**） | 例 **`…_n8_cuda_3arm_c12_20260411T0202Z.json`** |

### 3.4 机制分列（§7、L3 轨迹）

| 内容 | 登记 | 备注 |
|------|------|------|
| §7 玩具毫秒（S1–S4） | **`X-20260421-*`** | **单列**物理含义；**不可**与 path-batch 纵轴相减 |
| L3 轨迹甲·乙 | **`X-20260411-tf-kv-trajectory-l3-minimal`** | **≠ M1 全 DFS** |

---

## 4. 与「全模型 KV / Agent」的边界（成文必带）

- 本文 **path-batch** 与 **M1 玩具 TF-KV** 均 **不**承担 **全解码栈 KV 总账** 的宣称。  
- **SSGS / M1** 支持的是 **树导航 + 回溯代价结构** 的 **证据链**（**`RESEARCH_STATUS` §3.5 L1–L3**），**L4 级 Agent** 仅 **局限 / Future Work**（见 **`PHASE1_MANUSCRIPT` §9.2**）。

---

## 5. 相关文档（按阅读顺序）

1. **`PROJECT_MASTER_PLAN.md` §1.0** — 主验证轴表  
2. **`experiments/planning/SSGS_MAINLINE_M1.md`** — M1 工具链与 M2 跑道  
3. **`experiments/phases/FIGURE_CAPTIONS_STAGE1.md`** — 七轴防混读  
4. **`experiments/phases/FOUNDATION_STAGE_FORMAL_SUMMARY.md`** — 论文体例浓缩结论  
5. **`overview/execution/SUBMISSION_PACK.md` §A3** — 可粘贴边界段  
6. **`overview/execution/PLAN_NOW_TO_DONE.md` §Ⅶ** — **F0–F5** 可落地框架与阶段 5 关系  
7. **`PLAN_NOW_TO_DONE.md` §Ⅷ** — **论文 1 启动门闩**（工程闭环 + **真实 HF Transformer 臂** 可比）；**与「阶段 5 先成稿」二选一主轴**（见该文件篇首 **战略 A/B**）  
8. **`docs/research/RESEARCH_NOTES.md` §8** — **树状 RAG / 回溯文献** 与本文 **path-batch·SSGS·M1** **边界**（**RW 动机**，**不单开文件**）

---

## 6. 主线下一步（叙事已立 → 可执行推进）

> 与 **`PLAN_NOW_TO_DONE.md` §Ⅲ（可选 M2）**、**§Ⅵ（北星增量）** 对齐；**阶段 5 成文**仍为默认瓶颈。

| 优先级 | 动作 | 产出 / 完成标志 |
|--------|------|------------------|
| **P0** | 将 **§0.1 / §0.2** 迁入 **`PHASE1_MANUSCRIPT`** 或 LaTeX 摘要；与 **`SUBMISSION_PACK` §A1b** 对表 | **2026-04-11**：**`PHASE1_MANUSCRIPT`** **摘要 / §7 英文** 已同步 **§0**；终稿 LaTeX 再粘贴一次 |
| **P1** | **`§A2` basename** 终稿人工勾选（**`EXPERIMENT_REGISTRY`** 逐行） | 无「文中路径 / 登记册」不一致 |
| **Ⅲ-2**（可选） | 服务器 **`RUN_WIKITEXT_SMOKE=1`**（**`run_ssgs_mamba_wikitext_cuda.sh`**） | 同树 **path-batch + SSGS** 一行 JSON；**脚注分列**。**2026-04-11**：多次 **仅 SSGS n8**（含 **`…T0334Z.json`**），**未**带 smoke；grid **16** 行 — 若需 **bundle** 再开 **`RUN_WIKITEXT_SMOKE=1`** |
| **Ⅵ-1**（科研） | 在 **path-batch** 或 **M1** 槽位 **加规模**（例：**dim256 M1**、**XL 叶数**）— **新 STAMP**、**登记一行** | 结论**形状**是否仍支持「实现敏感 + 回溯代价结构」叙事；**禁止**与 **paper_main_v1** 无脚注混表 |

**公平口径**（主线对比）：**协议公平** + **测量轴分列** — 见前序 **协议公平** 定案；新实验须 **新脚注**。

---

## 7. 服务器运行命令（**Ⅲ-2**：同树 path-batch smoke + SSGS + 聚合）

**目的**：在 **Wikitext 同建树**上多写一行 **path-batch** JSON（`benchmark_wikitext_tree` **n8 c8 dim128**），再跑 **SSGS×Mamba** 与 **`ssgs_mamba_wikitext_grid.csv`**；与 **M1 DFS**、**path-batch 主文格点** **脚注分列**（**`FIGURE_CAPTIONS_STAGE1.md` 七轴**）。

**环境**：Linux + GPU；仓库已克隆；**`conda activate mamba2`**；详见 **`docs/environment/runbooks/AUTODL_SETUP.md`**。

在服务器上 **整段复制**（路径按实例改为你的仓目录；默认结果 **`$MAMBA2_RESULTS_ROOT/metrics_result/`**，常为 **`/root/autodl-tmp/mamba2_results/metrics_result/`**）：

```bash
cd /root/autodl-tmp/mamba2   # 或你的克隆路径
export HF_ENDPOINT=https://hf-mirror.com

RUN_WIKITEXT_SMOKE=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
```

**产出**：

- **`benchmark_wikitext_ssgs_bundle_<STAMP>_n8_c8.json`** — path-batch 三 reader 同树 smoke；  
- **`ssgs_mamba_wikitext_n8_c8_dim128_cuda_<STAMP>.json`** — SSGS 迹；  
- **`ssgs_mamba_wikitext_grid.csv`** — 通配合并（**`aggregate_ssgs_mamba_wikitext_json.py`** 已在脚本末尾调用）。

**回仓后**（Windows / 本机，仓库根）：把 **`metrics_result/*.json`** 与 **`ssgs_mamba_wikitext_grid.csv`** 拷入 **`results/metrics_result/`**，再执行：

```text
py -3 scripts/research/aggregate_ssgs_mamba_wikitext_json.py -g "results/metrics_result/ssgs_mamba_wikitext_*.json" --out-csv results/metrics_result/ssgs_mamba_wikitext_grid.csv
```

（Linux 上把 **`py -3`** 换成 **`python`**。）**登记**：**`EXPERIMENT_REGISTRY.md`** **`X-20260407-ssgs-mamba-wikitext-tree`** 附 **新 STAMP** 一句。

**仅重跑 SSGS、不要 path-batch smoke** 时：同上脚本但 **不设** **`RUN_WIKITEXT_SMOKE`**（默认 `0`）。

---

## 8. 「Mamba + 树状 RAG + SSGS」主线完成度（阶段 5）

**实证侧（本仓已可支撑投稿叙事）**

| 块 | 状态 | 备注 |
|----|------|------|
| **树 + path-batch + 三 reader** | **齐** | 主图 / **`paper_main_*`**；**naive vs fused** 分列 |
| **Wikitext 同 harness 扩展** | **齐** | 阶段 2 网格、登记见 **`DATA_ARCHIVE`** |
| **SSGS × Mamba × Wikitext（迹）** | **齐** | **`ssgs_mamba_wikitext_grid.csv`**；**N** = **`aggregate_*` stdout**（当前曾 **16**）；**勿**与 path-batch 墙钟混轴 |
| **同树 DFS 对照（M1）** | **齐** | **`ssgs_vs_kv_wikitext_nav_grid.csv`**；**玩具 TF-KV** 脚注 |
| **§7 机制分列** | **齐** | 与上 **各为独立轴** |

**成文侧（把主线「收口」为可投稿）**

1. **`PHASE1_MANUSCRIPT`**：摘要已含 **path-batch + SSGS/M1 一句 + 局限**；**§4 / §5** 与 **`FIGURE_CAPTIONS_STAGE1` 七轴** 一致。  
2. **`SUBMISSION_PACK` §A2–A3**：basename 与正文 **人工对表**。  
3. **不必**为「完成感」再无限次 **n8 SSGS** 重跑；新 **STAMP** 仅在有 **`git_sha` 刷新 / 审稿补格** 需求时加跑。

**仍可选的一条「拼图」**：**`RUN_WIKITEXT_SMOKE=1`** → 同树 **path-batch bundle** JSON（与 **SSGS 计数** 仍 **分列**脚注）— **不阻塞** 投稿。

**科研延伸（第二优先级）**：**`PLAN_NOW_TO_DONE` §Ⅵ-1**（加 **dim / 叶数** 在 **path-batch 或 M1** 槽位），与 **主线成文** 并行排期。

---

## 9. 实证数据快照（仓内；**N** 以 `aggregate_*` **stdout** 为准）

在**仓库根**重聚合后核对（**Linux** 用 **`python`**，**Windows** 常用 **`py -3`**）：

| 汇总 CSV | 数据行 **N** | 登记 id / 含义 |
|----------|--------------|----------------|
| **`results/metrics_result/ssgs_mamba_wikitext_grid.csv`** | **以当次 `aggregate_*` stdout 为准** | **`X-20260407-ssgs-mamba-wikitext-tree`** — **SSGS×Mamba×Wikitext** 迹（**非** path-batch 墙钟）。**N** 随通配到的 **`ssgs_mamba_wikitext_*.json`** 个数变（例：**16** → 下回多拷入文件后可为 **17**） |
| **`results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`** | **同上** | **`X-ssgs-vs-kv-tree-nav-m1`** — 同树 **DFS 三臂** + 多 **STAMP**（含 **c12** 等；**玩具 TF-KV** 脚注）。例：**18** → 同步更多 **`ssgs_vs_kv_tree_nav_wikitext_*.json`** 后可为 **21** |

**命令**（与 **`NEXT_EXPERIMENTS_COMMANDS.md` §12** 一致）：

```bash
cd /path/to/mamba2
python scripts/research/aggregate_ssgs_mamba_wikitext_json.py \
  -g "results/metrics_result/ssgs_mamba_wikitext_*.json" \
  --out-csv results/metrics_result/ssgs_mamba_wikitext_grid.csv
python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py \
  -g "results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json" \
  --out-csv results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv
```

**同树 path-batch + SSGS bundle（辅线 JSON）**：例 **`benchmark_wikitext_ssgs_bundle_*_n8_c8.json`**（**`RUN_WIKITEXT_SMOKE=1`**）；与上表 **分列**登记，**不计入** SSGS grid **行数定义**。

**`git_sha`**：服务器归档 JSON 常为 **`6fa7873`**；本机 **`git rev-parse --short HEAD`** 若在之后提交上开发，二者可不一致 —— **以 JSON 内字段为准**，刷新时按 **`PLAN_NOW_TO_DONE` §Ⅲ-3** 重跑单格。

---

## 10. 可落地「完整框架」：双轨路线图（投稿证据链 ↔ 北星）

> **说明**：**阶段 5** 已证明 **「Wikitext 真语料建树 + 同 harness path-batch + DFS 上 SSGS + M1 玩具对照」** 在**受控小宽度**下**机制闭环**。**「预训练 Mamba2 权重 + 真实下游任务 + 工程可部署」** 是 **同一方向上的升级档**，须 **新 harness / 新登记 / 新脚注**，见 **`RESEARCH_STATUS_AND_DIRECTION.md` §6.3–§6.4** 与 **`PLAN_NOW_TO_DONE` §Ⅵ–§Ⅶ**。

| 阶段 | 目标 | 本仓现状 | 下一跳 |
|------|------|----------|--------|
| **F0 冻结** | 叙事与数据可对表投稿 | **§8** 实证齐；**`EXPERIMENT_REGISTRY`** + **`DATA_ARCHIVE`** | **Ⅰ–Ⅱ**（**`PLAN_NOW_TO_DONE`**）：**§A2** basename、**pytest**、**commit** |
| **F1 公平协议** | 可比任务与预算维度写死 | **M1** 脚注已提醒跨臂墙钟不对等 | **Ⅵ-0** 一页纸；新实验 **新脚注** |
| **F2 规模抬高** | 结论「形状」在更大 dim/叶数上是否仍成立 | **path-batch leavescale**、**SSGS grid** 至 **n128**；**M1** **n64+L3** 已档 | **Ⅵ-1**：**M1 dim256** 或 **path-batch XL**（**`SSGS_MAINLINE_M1` §6.3**）；**新 STAMP** |
| **F3 预训 reader** | 路径编码用 **HF 预训 Mamba2**（或冻结 LM）而非纯随机初始化 toy trunk | **未**合入默认 harness；需 **新 reader 适配层 + `kind`** | **Ⅵ-2**：设计节 + 脚本名在 **`PROJECT_MASTER_PLAN`** 副线排期 |
| **F4 风险电池** | 有损回溯 / 回退策略 / 设备迁移 | **L3 minimal**、**§7 S4** 叙事已有 | **Ⅵ-3**：扩展 **`kind`**，与主表 **分列** |
| **F5 端到端系统** | 大上下文 LM + 树导航 + 训练/推理一体 | **超出** 本篇阶段 5 宣称 | **Ⅵ-4**：**48G**、协议先定；**审稿后 / 第二篇** 更现实 |

**「完整框架」最小可交付（工程视角）**：**建树模块**（Wikitext 叶块 → 平衡树）+ **reader 槽位**（Mamba / TF / GRU）+ **导航器**（**`dfs_ssgs_mamba`** + cache 快照）+ **JSON 迹与聚合 CSV** —— **F0 已覆盖**。**F3–F5** 把 **预训权重与业务任务** 接进同一槽位，属于 **产品线化** 增量，**不与** F0 的投稿混轴。

---

## 11. 推荐下一步：实验运行指令（按优先级）

**环境**（AutoDL）：**`conda activate mamba2`**；**`docs/environment/runbooks/AUTODL_SETUP.md`**；**`bash scripts/server/bootstrap_autodl.sh`**（首次）。

| 优先级 | 做什么 | 命令（服务器上 **整段**；路径改你的仓） |
|--------|--------|----------------------------------------|
| **1 成文** | **不跑 GPU** | **`SUBMISSION_PACK` §A2–A3**；全文 basename ↔ **`EXPERIMENT_REGISTRY`** |
| **2 仓内核对** | 重聚合 + 测试 | 见 **§9** 两条 **`aggregate_*`**；**`python -m pytest tests/ -q`** |
| **3 可选 · 同树 bundle** | path-batch smoke + SSGS 一行 | **`RUN_WIKITEXT_SMOKE=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh`**（**§7**） |
| **4 可选 · 仅 SSGS n8** | 刷新 **`git_sha`** 或补 STAMP | **不设** **`RUN_WIKITEXT_SMOKE`**：`bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh` |
| **5 可选 · M1 全链** | 三臂 DFS + 叶扫 / L3 | **`bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**；**`M1_LEAVES="8 16 32 64"`** 等见 **`NEXT_EXPERIMENTS_COMMANDS.md` §10.1** |
| **6 北星 · 规模** | **Ⅵ-1** 一格 | **`M1_LEAVES="8"`** + **`M1_STAMP=$(date -u +%Y%m%dT%H%MZ)`** 或 **`SSGS_MAINLINE_M1` §6.2 B3**（**`--chunk-len 12`** 已档） |

**回 Windows 入仓**：**`scp`/网盘** → **`results/metrics_result/`** → 再跑 **§9** 聚合 → 更新 **`EXPERIMENT_REGISTRY`** / **`DATA_ARCHIVE`**。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：叙事链 + 数据索引表 + 边界句；与 **`PROJECT_MASTER_PLAN` §1.0** 互链 |
| 2026-04-11 | **§0**：中/英摘要可用稿 + 关键词；**§6** 主线下一步（P0 / Ⅲ-2 / Ⅵ-1） |
| 2026-04-11 | **§7**：**Ⅲ-2** AutoDL 一键命令；**P0** 摘要已并入 **`PHASE1_MANUSCRIPT`** |
| 2026-04-11 | **§6**：**0303Z** **n8** SSGS 复跑 → **grid 15** 行；登记 **`X-20260407`** 脚注；**Ⅲ-2** smoke **仍可选** |
| 2026-04-11 | **§8**：**Mamba+树+RAG+SSGS** 主线 **完成度表**（阶段 5）；**0334Z** → grid **16** 行 |
| 2026-04-11 | **§9**：聚合快照 **SSGS N=16**、**M1 N=18**；**§10** 双轨 **F0–F5**；**§11** 命令优先级表 |
| 2026-04-11 | **§9**：**N** 改为 **仅以 stdout 为准**（下回同步 **`metrics_result`** 后例 **17 / 21**） |
| 2026-04-11 | **§5**：互链 **`PLAN_NOW_TO_DONE` §Ⅷ**（**论文 1 门闩** / **战略 B**） |
| 2026-04-11 | **§5**：**`RESEARCH_NOTES` §8**（**RW·树状 RAG/回溯** 与本文边界） |
