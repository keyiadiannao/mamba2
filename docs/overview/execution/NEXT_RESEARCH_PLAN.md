# 下一步研究计划（展开）

> **先读**：**`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`**（整体方向、现状、**§3.5**、决策原则、**§6**、与本文件 **推荐顺序** 的对应）。  
> **前置**：阶段 1 已收束（**`docs/experiments/phases/PHASE1_MANUSCRIPT.md`**、**`results/metrics_result/`**、§7 复跑验收）。本文将 **`docs/overview/planning/PROJECT_MASTER_PLAN.md`** 工作分解 **展开为可执行任务**（**`docs/overview/planning/ROADMAP.md`** 仅保留 **阶段 2 入口指针**）；**周期勾选**仍以 **`docs/overview/execution/CURRENT_SPRINT.md`** 为准。  
> **2026-04 服务器数据索引**（JSON / STAMP / CSV）：**`docs/experiments/planning/DATA_ARCHIVE_202604_SERVER.md`**。  
> **阶段 0→结题**（每阶段 **实验 + 成功标准**；**当前 = 阶段 5 成文**）：**`docs/overview/planning/RESEARCH_PHASES_0_TO_DONE.md`**。

---

## 0. 全局阶段、路线图与下一步（2026-04-11）

### 0.1 当前所处阶段（一句话）

**阶段 1–2 的实证与登记已基本收口**：**path-batch**、**§7**、**A2-S3**、**SSGS（grid 13 行）**、**Phase M1**（三臂 + L3；nav grid **N** 数据行 + 表头，**N** 见 **`aggregate_ssgs_vs_kv_wikitext_json.py` stdout**）、**B-S2+ CUDA**、**玩具 L3 轨迹甲·乙（`tf_kv_trajectory_l3_minimal`）** 等均 **入仓 JSON / CSV** 并与 **`EXPERIMENT_REGISTRY`** 对齐。**当前主瓶颈** 已从「补跑服务器格点」转为 **P0：成文冻结与投稿叙事**（**`SUBMISSION_PACK` §A1–A4** 粘贴正稿、**七轴**脚注不混读）。  
**方向性说明**：**P0 收口 ≠ 研究终点**。长期叙事仍以 **`RESEARCH_STATUS_AND_DIRECTION.md` §1.5**（**状态快照回溯 × 树状 RAG** 北星）与 **§3.5**（证据层级 / 风险 / PoC）为准；阶段 1–2 是为该北星服务的 **可发表地基**，**L4 级 Agent 声称** 须 **另排期、另 harness**。

### 0.2 从现在到远期的分阶段计划

| 阶段 | 名称 | 目标 | 主要产出 |
|------|------|------|----------|
| **P0** | **投稿前成文冻结** | 数字/路径与登记册一致；主文+附录脚注齐备 | **`PHASE1_MANUSCRIPT`** 定稿段落；**`SUBMISSION_PACK` §A2–A3** 入稿；可选 **§7.5 S5** 视篇幅 |
| **P0+** | **截稿后修订 / 审稿回应** | 按意见补表、补一句边界、或补 **1 格** smoke | 修订版 JSON（新 **`git_sha`**）、**`EXPERIMENT_REGISTRY`** 新行 |
| **阶段 2 延伸（科研）** | **更大语料 / 更深树 / XL 复跑** | 在 **不推翻 harness** 下扩展 **A2** 网格或 **path-batch XL** | 新 **`kind` 或 STAMP**；仍须 **5060/3090 分列** |
| **机制深化** | **B-S3、训练型 L3** | 探针在更大 reader 上是否保持；**M1** 外 **训练子头** 另 **`kind`** | **`RETRIEVAL_HEAD_NOTES`**、新探针 JSON |
| **另立项** | **L4 / Agent 叙事** | 仅当证据链从 **L3** 自然延伸 | 新路线图文档；**不**在现阶段主文跳级写成 Agent |

### 0.3 立即下一步（具体一条）

**本机 / 任意环境**：**`git add`** 已入仓的 **`results/metrics_result/*.json`**（若尚未提交）+ 本轮 **docs** 修订 → **`git commit`**。**成文**：打开 **`PHASE1_MANUSCRIPT.md`**，将 **`SUBMISSION_PACK.md` §A3（七轴脚注）** 与 **§A2.1** 的边界句 **粘贴进正文/附录** 对应小节，并 **对照 `DATA_ARCHIVE_202604_SERVER.md` §0** 核对 basename。**Linux / torch 可用**：**`pytest tests/ -q`** 全量计数并回填本文 **§1** 与 **`SUBMISSION_PACK` §A8** 中的 **passed** 数字。

---

## 项目现状快照

**主线材料、完成度大表、七轴防混读**：**唯一权威** **`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md` §2–§3**（不在此重复表格）。**本机可复制命令**：**唯一权威** **`docs/environment/runbooks/LOCAL_5060_RUNBOOK.md`**。

### 1. 测试与仓库卫生

- **`python -m pytest tests/ -q`**（须 **`mamba2`** 等有 **torch** 的环境）：**AutoDL 实测** **28 passed**, **4 subtests**, **~21 s**（以你当次输出为准）。
- **`py -3 -m pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py -q`**：**无 torch** 亦可跑（**2** 条），见 **`LOCAL_5060_RUNBOOK` §5**。
- **`git status`**：提交前自检应保持 **干净**；勿手改污染 **`metrics_result/`** 归档。

### 2. **3090 / AutoDL** 批次：**已完成的登记项**（归档核对）

**（以下三条原属 **P1/P2**；**2026-04-10** 前后已入 **`results/metrics_result/`** 与 **`EXPERIMENT_REGISTRY`**。新跑次仍按 **`NEXT_EXPERIMENTS_COMMANDS.md`** 执行并 **新开登记行**。）**

1. ~~**B-S2+ CUDA**~~：**`results/metrics_result/probe_path_reader_linear_text16_heldout_train50_cuda_20260410T1302Z.json`**；登记 **X-20260410-probe-path-reader-bs2plus-cuda-3090**（与 **`LOCAL_5060_RUNBOOK` §2** CPU 分列）。  
2. ~~**SSGS n8 刷新 `git_sha`**~~：**`20260410T1238Z`** 等同批已含 **n8**（见 **X-20260407-ssgs-mamba-wikitext-tree**）。  
3. ~~**SSGS n128**~~：**`ssgs_mamba_wikitext_n128_c8_dim128_cuda_20260410T1238Z.json`** + grid **13 行**。

---

## 3. 正式开工：**Mamba + 树 + SSGS** 整合主线（**Phase M1**）

> **唯一详表**：**`docs/experiments/planning/SSGS_MAINLINE_M1.md`**（工具盘点、缺口、四周目标、检查表）。  
> **登记**：**`EXPERIMENT_REGISTRY.md`** **`X-ssgs-vs-kv-tree-nav-m1`**。  
> **Harness（已落地）**：**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`**（**`kind=ssgs_vs_kv_tree_nav_wikitext`**）— **Mamba** + **TF-KV clone** + **TF-KV truncate_kv**；**`run_m1_ssgs_vs_kv_wikitext_cuda.sh`**；可选 **`--l3-tf-kv-hidden`** / **`--l3-tf-kv-downstream-ce`**。  
> **归档摘要**：多 **STAMP** 的 **`ssgs_vs_kv_tree_nav_wikitext_*.json`**；汇总 **`results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`**（**`aggregate_ssgs_vs_kv_wikitext_json.py`**）。**L3** 与 **n8/n16/n32** 实测见 **`SSGS_MAINLINE_M1.md`** §2.1。

**已完成（相对 2026-04-11 稿）**：**同 Wikitext 建树、同 DFS 任务**下的 **三臂统一 JSON**、**叶扫至 n64**（**三臂**，例 **`…_n64_cuda_3arm_20260410T1235Z.json`**）、**隐状态 L3**（**n8** 等）、**固定叶头 CE（n8 / n16 / n32 / n64 已归档**，同批例 **`STAMP=20260410T1247Z`** 四 JSON）、**网格 CSV**（**数据行 = 聚合 `N row(s)`** + 表头）与 **登记册** 更新。**M2 · B2** 在 **本仓已齐** 时 = **AutoDL 上复现 / `git pull` 后刷新 `git_sha` 与 `abs_ce_delta` 自检**（见 **`SSGS_MAINLINE_M1` §6.0–§6.2**）。  
**可选后续**：**`git pull` 后** 单点 smoke 刷新 **`git_sha`**；**训练型子头 / 树 LM 对齐** 须 **另 `kind`**。**~~轨迹甲·乙（玩具 TF-KV）~~** 已归档：**`tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`**（**`X-20260411-tf-kv-trajectory-l3-minimal`**）。**成文**：在 **`FIGURE_CAPTIONS_STAGE1.md`** / **`PHASE1_MANUSCRIPT` §5.1** / **`SUBMISSION_PACK` §A2–A3** 写入 **M1** 与 **七轴**（含 **L3 轨迹**），**禁止**无脚注混读。

**与 P0 关系**：**P0 成文** 为主。**原 P1/P2**（B-S2+ CUDA、SSGS n128）与 **阶段 C L3 轨迹 JSON** 均已归档。**可选**：path-batch **XL** 复跑、**B-S3** 检索头 — 见 **`RESEARCH_STATUS` §3.5**、**`NEXT_EXPERIMENTS_COMMANDS` §0.5**。

### 3.1 **M2** 后续实验（**Mamba + 树 + SSGS** 在 **M1** 归档之后）

**唯一详表与命令**：**`docs/experiments/planning/SSGS_MAINLINE_M1.md` §6**（**§6.0** 说明 **B2** 与 **「全链条」①②③④** — **B2 只补环节③ 在 n64 的 L3 CE 列，不是从零验证全链，也不是新对比法**）。Wave **A** 成文并行；**B** 云端 **B1–B4**；**C** 延伸；**D** = P★。**AutoDL**：若要先确认 **M1 链** 可实现，**先** **`M1_LEAVES="64"`** 三臂 **无 L3**，**再** **B2**（见 **§6.0** 两步）。

---

## 后续方向（推荐优先级）

| 优先级 | 内容 | 说明 |
|--------|------|------|
| **P0** | **成文整合** | **`PHASE1_MANUSCRIPT`** / **`FIGURE_CAPTIONS_STAGE1`** / **`EXPERIMENT_REGISTRY`** 对齐投稿版；**§7.5 S5** 总表 **视截稿篇幅** |
| **M1** | **SSGS 整合对照（主线科研）** | **`SSGS_MAINLINE_M1.md`**：**harness 已归档**（**`X-ssgs-vs-kv-tree-nav-m1`**）；**L3** 见 **`RESEARCH_STATUS` §3.5**；成文须 **单列脚注**（**`FIGURE_CAPTIONS_STAGE1.md`** **第七轴：M1 DFS 三臂**） |
| ~~**P1**~~ | ~~**3090：B-S2+ CUDA**~~ | **已完成**：**`probe_path_reader…cuda_20260410T1302Z.json`**；登记 **X-20260410-probe-path-reader-bs2plus-cuda-3090** |
| ~~**P2**~~ | ~~**SSGS n128 等**~~ | **已完成**：**n128** 等在 **`ssgs_mamba_wikitext_grid.csv`（13 行）**；登记 **X-20260407-ssgs-mamba-wikitext-tree** |
| **P3** | **A2-S3 可选加压** | **云端或本机 CPU**（视脚本）；与 **init×5** **分列** 说明 |
| **P★** | **（可选）训练型 L3** | **M1** 已含 **隐状态 + 固定头 CE**；**玩具轨迹** 已 **`tf_kv_trajectory_l3_minimal`**；**训练型探针** 另 **`kind`**；**`RESEARCH_STATUS` §3.5** |

**默认里程碑顺序（2026-04-11 起）**

1. **P0 成文** 与 **M1 叙事入稿**（**七轴脚注**）**并行**。  
2. **M1 可选实测**：**`git pull` 后** 单点 smoke 刷新 **`git_sha`**；**不**阻塞 **P0** 冻结。  
3. **3090 有空时**：仅 **可选** 新格点（**path-batch XL**、**B-S3** 等）；**不**阻塞投稿。  
4. **P3**、**P★**（**训练型 L3**）仍 **可选**。

**原则**：**不**在无脚注下混表 **5060 naive** 与 **3090 fused**；**不**混读 **path-batch 毫秒/峰值**、**§7 单列毫秒**、**SSGS 快照计数**、**A2-S3 准确率**、**M1 三臂 DFS 对照**、**L3 轨迹玩具 KV**（**`PHASE1_MANUSCRIPT` §5.1** + **`FIGURE_CAPTIONS_STAGE1.md`** **七轴表** + **`SSGS_MAINLINE_M1.md`**）。**§3.5** 证据层级：**M1** 目标 **L3**，**不**跳级写成 L4 Agent。

---

## 算力不可用时的备选推进（成文 + 本机可执行）

**定位**：**仅当 AutoDL/3090 暂时不可用时** 使用；按 **`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md` §4–§5** 与 **`PROJECT_MASTER_PLAN`** 的 **正常节奏** 推进；**不**把 **P★** 插入本段。**服务器可用时** 以 **§0.3 / 「服务器有空时」** 与 **`docs/environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md`** 为主（**§2** 为 **已完成项** 核对）。

### A. 成文（P0，优先）

**成文包（A1–A4 正文与脚注草稿）**：**`docs/overview/execution/SUBMISSION_PACK.md`**（一页故事线、路径核对表、可粘贴边界句、检索头短段；**§A1b** 摘要/引言草稿）。

| 顺序 | 任务 | 产出/自检 |
|------|------|-----------|
| **A1** | **主叙事一页** | **`SUBMISSION_PACK.md` §A1**；摘要/引言句见 **§A1b** |
| **A2** | **数字与路径核对** | **`SUBMISSION_PACK.md` §A2** + 全表补扫 **`PHASE1_MANUSCRIPT` §5.1** |
| **A3** | **七轴脚注成句** | **`SUBMISSION_PACK.md` §A3**；完整表见 **`FIGURE_CAPTIONS_STAGE1.md`** |
| **A4** | **检索头边界** | **`SUBMISSION_PACK.md` §A4**；全文见 **`PHASE1_MANUSCRIPT` §9** |
| **A5** | **（可选）§7.5 S5** | 视截稿；**非**阻塞 |

### B. 本机可补的实验（**非**必须；仅当成文需要「附录多一行」）

**环境**：**`docs/environment/runbooks/LOCAL_5060_RUNBOOK.md`**；解释器 **`mamba2`**。**登记**：新 JSON 须 **`docs/experiments/planning/EXPERIMENT_REGISTRY.md` 新行**。

| 顺序 | 内容 | 说明 |
|------|------|------|
| **B1** | **`probe_retrieval_correlation.py --cpu`** | **B-S2** 附录；**`RETRIEVAL_HEAD_NOTES.md` §2**；与 path-batch **分列** |
| **B2** | **`benchmark_mamba2_cache_snapshot_segments.py --device cpu`** | §7 **S1** 本机复现/更新 JSON（与 **`docs/environment/runbooks/SERVER_SWEEP_RUNBOOK.md`** 协议一致） |
| **B3** | **`python -m pytest tests/ -q`** | **28 passed + 4 subtests**（**AutoDL** 样例；以当次为准）；提交前 **smoke** |

**已跑满时不必重复**：本机 **B-S2+**、**A2-S3 n8**、**Wikitext CPU/CUDA smoke**、**SSGS 轻量** 等见 **「本机 5060 已完成」** 清单。

### C. 算力恢复后

**P1/P2**（B-S2+ CUDA、SSGS n128 等）**已归档**；算力恢复后仅 **可选** 刷新 **`git_sha`**、**XL**、**B-S3** —— 见 **§0.2** 与 **「服务器有空时」**。

---

## 当前收口清单（工作台整理；与 **`PHASE1_MANUSCRIPT.md` §10** 对齐）

**成文（优先，不占 GPU）**

- [x] **归档路径核对**：见 **`PHASE1_MANUSCRIPT.md` §5.1** + **`SUBMISSION_PACK.md` §A2**（主图 PNG、**`paper_main_*` CSV**、§7、5060、阶段 2 **含 `dim256` `0847Z`**、A2-S3 **含 TSV**、SSGS **13 行 grid**、**M1** **`ssgs_vs_kv_wikitext_nav_grid.csv`**）；七轴图注见 **`FIGURE_CAPTIONS_STAGE1.md`**。
- [x] **分列脚注规则**：已写入 **`PHASE1_MANUSCRIPT` §5.1**（**5060/3090**、**naive/fused**、**path-batch / §7 / SSGS / A2-S3**）；正式投稿前将对应句式 **粘贴进论文正文/附录** 即可。
- [ ] **§7.5 S5** 总表是否补 —— **视截稿篇幅**（主图 PNG **已入仓** 三张 **`mamba_3090_naive_vs_fused_dim{128,256,384}_*.png`**）。

**仓库与数据 hygiene**

- [x] **`git status`** 干净（提交前自检；更新本快照时 **已验证**）；勿让 Excel/手改 **污染** 已归档 **`benchmark_wikitext_stage2_*`** / **`…leavescale_xl_*`**（误改用 **`git restore results/metrics_result/<file>`** 回滚）。
- [x] **SSGS Wikitext**：以 **`ssgs_mamba_wikitext_grid.csv`** + **`ssgs_mamba_wikitext_*.json`** 为准；**勿**保留 **`…grid-Copy1.csv`** 等重复副本。

**本机 5060 已完成（2026-04-10；与登记册、`PHASE1_MANUSCRIPT` §4/§5/§8.2 对齐）**

- [x] **A2-S3**：**`task_wikitext_path_pair.py`** **`--cpu --num-leaves 8`**，**sibling**、**stratified**、**c8 dim128** → **`results/metrics/task_wikitext_sibling8_local5060_cpu_20260410.json`**（**X-20260410-local5060-a2s3-n8-strat**；**`ridge_concat.*.test_acc`≈0.857**，7 test 对）。
- [x] **SSGS × Wikitext（轻量）**：**`demo_ssgs_mamba_wikitext.py`** **`--cpu`** **n8 c4 dim64** → **`results/metrics/ssgs_mamba_wikitext_n8_c4_d64_local5060_20260410.json`**（**X-20260410-local5060-ssgs-wikitext-n8-c4d64**；**snapshots 7 / rollbacks 11 / leaf_checks 8**）。与 **`metrics_result` 归档 grid（c8 dim128）** **分列**。
- [x] **B-S2+ BCE 50 步（本机）**：**`probe_path_reader_linear_text16_heldout_train50_local5060.json`**（**X-20260410-local5060-bs2plus-train50-n16**；与 **`LOCAL_5060_RUNBOOK` §2** 可选行一致）。
- [x] **Wikitext path-batch CPU smoke**：**`benchmark_wikitext_local5060_cpu_20260410T1220Z_n8_c8.json`**（**`WARMUP=1` `REPS=2`**；**X-20260410-local5060-wikitext-cpu-n8c8**）。
- [x] **回归**：**`pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py`**（**2** passed，**`py -3`** 可无 torch）；**`python -m pytest tests/ -q`** 全量（**AutoDL**：**28 passed**, **4 subtests**, **~21 s**）。

**服务器有空时（可选一条即可；建议优先级自上而下）**

1. **`git pull` 后** **M1 或 SSGS** **单格 smoke** 刷新 **`git_sha`**（**非**投稿阻塞）。  
2. **path-batch XL** 或 **§7.5 S5** 所需 **新脚本/合成表**（**视截稿**）。  
3. **B-S3** 或 **训练型 L3**（**新 `kind`**）—— **阶段 2+**；见 **`RESEARCH_STATUS` §3.5**。

**已就绪（无需再跑也能写）**：阶段 2 **path-batch**（**A2-S2**、**dim256 `1137Z`+`0847Z`**、叶数扫描与 XL、**§7 depth 5–6**）、**A2-S3**（**1438Z** + **0820Z/0850Z** + **TSV**）、**Wikitext SSGS**（**`ssgs_mamba_wikitext_grid.csv` 13 行**）、**B-S2+ CUDA**、**L3 轨迹 JSON** —— 见 **登记册**、**`DATA_ARCHIVE_202604_SERVER.md` §0**、**`SUBMISSION_PACK` §A2/A2.1/A8**、**`PHASE1_MANUSCRIPT` §5**。

**M1 开工清单**（与 **§3** 一致）：见 **`docs/experiments/planning/SSGS_MAINLINE_M1.md` §4**。

**本机 5060 可复制命令**（含 B-S2+ / Wikitext / SSGS / §7 CPU）：**唯一权威** **`docs/environment/runbooks/LOCAL_5060_RUNBOOK.md`**；**勿**在此文件维护第二份命令表。

---

## 1. 总目标（接下来 4–8 周）

在 **不推翻阶段 1 harness** 的前提下：

1. **系统线（A）**：真语料 **浅层树** + **同一 path reader 三对比** + **至少 1 个任务级指标**，使论文从「纯效率曲线」推进到「树 + 读者 + 简单任务」。  
2. **机制线（B）**：检索头 **探针 / 相关性分析**（可先在小模型或冻结主干上），产出 **可进附录** 的图表或表。  
3. **叙事与协议线（X）**：主文 **Figure 1** 定稿（含 PNG 是否入仓）；**§7.5 S5** 是否在截稿前补一行「同等试错轨迹」对照 — **视篇幅** 取舍。

三条线可 **并行**，硬依赖仅：**A 的 harness 稳定** 后，B/C 的「树上读路径」实验才好对齐叙事。

---

## 2. 轨道 A：阶段 2 — 真语料浅树 + 任务指标

### 2.1 原则

- **硬约束**：新树必须 **进入现有** `run_tree_reader_benchmark` / `benchmark_wikitext_tree.py` **同接口**；否则另开脚本须在 **EXPERIMENT_REGISTRY** 单列一行并声明 **与阶段 1 不可比**。  
- **软目标**：先 **Wikitext 扩展**，再考虑 **第二语料**（`prepare_leaves_from_corpus.py` + `benchmark_text_tree.py`）或 **层次聚类 / RAPTOR 式** 建树（仅当能导出 **平衡或规范化的遍历序**）。

### 2.2 里程碑（建议顺序）

| 代号 | 内容 | 产出 | 机器 |
|------|------|------|------|
| **A2-S0** | **登记占位**：新开 **EXPERIMENT_REGISTRY** 行（建议 id：`A-stage2-wikitext-grid-v1`），写清 **dim / fanout / chunk / num_leaves 上界 / HF 镜像** | 一行登记 + 可选 `TAG` 名 | — |
| **A2-S1** | **Smoke**：`benchmark_wikitext_tree.py` 固定 **num_leaves≤16**、**depth≤4**、与 **A-20260408-wikitext** 同 **dim=128**，输出 **JSON**（含 `git_sha`、torch 版本） | `results/metrics_result/benchmark_wikitext_*_stage2_smoke.json` | 5060 或 3090 |
| **A2-S2** | **小网格**：默认 **四格** `{8,16}×{8,12}`（与 5060 naive 动机同拓扑），**fused**；**`WARMUP`/`REPS`** 默认与 **paper_main** 一致（可 **`REPS=5`** 对齐 5060） | JSON + 汇总 CSV + manifest；**`SERVER_SWEEP_RUNBOOK.md` §2d** + **`run_server_stage2_wikitext_grid.sh`** | AutoDL |
| **A2-S2b** | **叶数扫描**：固定 **chunk_len**、**dim=128**，**`num_leaves ∈ {8,16,32,64}`**；**`run_server_wikitext_leavescale.sh`**（**`SERVER_SWEEP_RUNBOOK` §2f**）；登记 **A-stage2-wikitext-leavescale-v1** | 同 **A2-S2** harness；**TF** 为 **O(T²)** 整段 SA | AutoDL |
| **A2-S3** | **任务指标 +1**（择一落地）：<br>• **浅层路径分类**：给定叶对 / 节点对，预测是否同子树（需 **自动生成标签** 脚本）；<br>• **固定句填空 / 选词**：用叶块文本构造 **cloze**，读路径后 MLP 头预测（最小可用 **tiny LM 或池化+logreg**）；<br>• **检索式**：query → 哪片叶最相关（小候选集上的 **top-1 acc**）。 | **最小实现（v0）**：**`task_wikitext_path_pair.py`**（叶对 **同 cohort**、ridge on **concat(z_i,z_j)**；**`--pair-split leaf_heldout`** 减叶对泄漏；登记 **A-20260407-stage2-wikitext-path-pair**；**`PHASE2_DRAFT.md`** §2）。其余选项仍可用 notebook 另开。 | 48G 优先；**CPU/smoke** 可跑 v0 |
| **A2-S4** | **成文段落**：在 **`PHASE1_MANUSCRIPT.md`** 后续或 **`docs/experiments/phases/PHASE2_DRAFT.md`** 补 **半页「真语料 + 任务」**；图注沿用 **`FIGURE_CAPTIONS_STAGE1.md`** 边界句式 | 文档 PR | — |

### 2.3 技术注意

- **叶块长度不一**：须在登记中固定 **tokenizer 截断长度** 与 **节点嵌入构造**（当前为确定性 hash 嵌入则写明 **非神经 encoder**）。  
- **与合成树对比**：正文写 **「同 harness、不同语料」**，避免把 Wikitext 点与 paper_main 合成点 **叠在同一张绝对坐标图** 除非 **归一化轴** 说明清楚。

### 2.4 建议首条命令（Smoke）

**`benchmark_wikitext_tree.py`** 默认仍 **打印 JSON 到 stdout**；归档时请使用 **`--out-json PATH`**（与 **`demo_tree_lm_minimal.py`** 一致写入 **`git_sha`**、**`torch_version`**）：

```bash
# 仓库根，conda activate mamba2；HF 不通时：
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p results/metrics_result
STAMP=$(date -u +%Y%m%dT%H%MZ)

python scripts/benchmarks/benchmark_wikitext_tree.py \
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 \
  --warmup 2 --reps 5 \
  --out-json "results/metrics_result/benchmark_wikitext_stage2_smoke_${STAMP}.json"
```

（仍可用 shell 重定向代替 **`--out-json`**，但推荐后者以便 **父目录自动创建** 与 **与 stdout 同一份** 校验。）

**5060 CUDA 四格汇总表（本地）**：**`scripts/benchmarks/aggregate_wikitext_5060_cuda_grid.py`** → **`results/metrics_result/benchmark_wikitext_5060_cuda_grid_20260407.csv`**（与 **`PHASE2_DRAFT.md` §1.1** 配套）。

---

## 3. 轨道 B：检索头分析（`PROJECT_MASTER_PLAN` B）

### 3.1 目标

在 **固定小树或扁平块** 上，验证 **哪些层/头** 对「是否需要检索」或「路径位置」敏感，为后续 **注入（C）** 提供 **假设**。

### 3.2 里程碑

| 代号 | 内容 | 产出 |
|------|------|------|
| **B-S1** | 文献与设计空间 **半页**（Hidden Attention / RAD 等 **引用 + 本文差异**） | `docs/research/RETRIEVAL_HEAD_NOTES.md`（新建） |
| **B-S2** | **探针脚本**：对 **HF 小因果 LM** 或 **path reader 表征** 抽层向量，算与 **合成 / 随机标签** 的 **岭线性可分性**（对照 **random_label_control**） | `scripts/research/probe_retrieval_correlation.py` + **`--out-json`** |
| **B-S3** | **48G 窗口**：换 **更大 reader 或更深树** 复测 **趋势是否保持** | 登记新行 **B-stage2-probe-*** |

**依赖**：可与 **A2-S1** 并行；**不必**等任务指标 A2-S3。

---

## 4. 轨道 X：成文、图与 §7.5 S5

| 任务 | 说明 |
|------|------|
| **主图 PNG 入仓** | 若仓库策略允许，**`git add`** `mamba_3090_naive_vs_fused_dim*.png`；过大则 **网盘 + registry 外链** |
| **S5 汇总表** | `RESEARCH_NOTES` §7.5 **S5** 行：在 **同一 DFS 轨迹** 下对比 **SSM restore** vs **TF-R1** vs **TF-KV** 的 **wall-clock** — 需 **新脚本** 或 **把现有 JSON 合成一张表**；**截稿前**再判断是否值得 |
| **辅线 X-20260422–25** | 仅当审稿或导师要求 **「导航故事」** 时再加笔；**默认不增投** |

---

## 5. 依赖关系（简图）

```mermaid
flowchart LR
  A2_S0[A2-S0 登记]
  A2_S1[A2-S1 Wikitext smoke]
  A2_S2[A2-S2 小网格]
  A2_S3[A2-S3 任务指标]
  A2_S4[A2-S4 成文]
  B_S1[B-S1 笔记]
  B_S2[B-S2 探针]
  A2_S0 --> A2_S1 --> A2_S2 --> A2_S3 --> A2_S4
  B_S1 --> B_S2
```

---

## 6. 建议的最近两周

**唯一维护处**：**`CURRENT_SPRINT.md`**（与本文件同目录）。请勿在本节复制粘贴周任务（曾与 sprint **双写**）。

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-07 | **docs 目录重组** + **3090 可用**：`environment`→**runbooks/**·**troubleshooting/**，`overview`→**planning/**·**execution/**，`experiments`→**planning/**·**phases/**；原 **「无云端时」** 更名为 **「算力不可用时的备选推进」**；默认 **P0 成文** 与 **P1/P2（3090）** 可并行 |
| 2026-04-09 | 初版：阶段 2 / 检索头 B / 成文与 S5 展开 |
| 2026-04-10 | **B-S2**：`probe_retrieval_correlation.py`；**RETRIEVAL_HEAD_NOTES** §2 文献入口表（2404.15574 / 2406.15765 / 2209.11895 / 2508.02184 / 2410.10819） |
| 2026-04-10 | **B-S2+**：`--label-mode topic`、`--topic-split heldout`（模板留出，避免句级划分虚高）；gpt2 归档 `probe_retrieval_linear_gpt2_topic_{heldout,sample}_cpu.json` |
| 2026-04-10 | **B-S2+ path reader**：`probe_path_reader_linear` 16 叶 heldout + 可选 BCE；**`--train-head-only`** 与全量微调分列 JSON |
| 2026-04-10 | **A2 本地**：5060 CUDA **`benchmark_wikitext_tree`** `n16×c8` / `n16×c12` 入 **`results/metrics_result/`**（与 3090 **不可混点**） |
| 2026-04-07 | **A2-S3 v0**：**`task_wikitext_path_pair.py`**（Wikitext 叶对同 cohort + ridge）；**`PHASE2_DRAFT.md`**；登记 **A-20260407-stage2-wikitext-path-pair** |
| 2026-04-07 | **A2-S3**：**`--pair-split leaf_heldout`**（train/test 叶不交）；归档 **`task_wikitext_sibling16_leafheldout4_{cpu,cuda5060}.json`** |
| 2026-04-07 | **§2.4**：**`aggregate_wikitext_5060_cuda_grid.py`**；**`path_pair_geometry` + test**；**A2-S3** `chunk_len=12` leaf_heldout |
| 2026-04-07 | **A2-S2 云端包**：**`run_server_stage2_wikitext_grid.sh`** + **`aggregate_wikitext_tree_json_grid.py`**（**`--base-dir`**）；**`SERVER_SWEEP_RUNBOOK` §2d** |
| 2026-04-07 | **A2-S2b** 叶数扫描 **`run_server_wikitext_leavescale.sh`**；§7 **depth 5–6** **`run_server_section7_depth_sweep.sh`**（**`SERVER_SWEEP_RUNBOOK` §2f–§2g**） |
| 2026-04-09 | **A2-S2b** 已归档：**`TAG=stage2_leavescale`** **`STAMP=20260409T1257Z`**；**`EXPERIMENT_REGISTRY` A-stage2-wikitext-leavescale-v1** |
| 2026-04-09 | **A2-S2b XL**：**`stage2_leavescale_xl`** **n128/n256** **`1322Z`/`1324Z`**；**A-stage2-wikitext-leavescale-xl-v1** |
| 2026-04-10 | **§3.5 指针**；**「后续方向」** 增 **P★ L3 语义 PoC**、**已决策略**（主线保底 / 叙事升级可选 / 动机≠证据）；见 **`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md` §3.5** |
| 2026-04-10 | **（旧称「无云端时：标准推进」）** **A 成文**（A1–A5）+ **B 本机可选**（B1–B3）；**默认里程碑** **P0→P1→P2→P3**，**P★** 不插入；**`LOCAL_5060_RUNBOOK`** / **`CURRENT_SPRINT`** 对齐 |
| 2026-04-10 | **`SUBMISSION_PACK.md`**：**A1–A4** 成文包初版（故事线、路径核对、脚注句、检索头短段） |
| 2026-04-10 | **`SUBMISSION_PACK.md`**：**§A1b** 摘要/引言；**§A2** 主文 6 CSV + 5060 grid 等 **存在性 ✅** |
| 2026-04-10 | **备选推进 §B**（旧称无云端 §B）：本机 **§7 S1** + **B-S2 gpt2 topic heldout** 复跑 JSON；**`SUBMISSION_PACK`** **提交前检查**；**`LOCAL_5060_RUNBOOK` §5.1** |
| 2026-04-10 | **P0**：**`SUBMISSION_PACK` A5–A7** 成文骨架与搁置表；**§B**：本机 **§7 S2 TF-R1 CPU** 确认 JSON |
| 2026-04-10 | **投稿版对齐**：**`SUBMISSION_PACK` A2 扩表** + **A2.1/A8**；**`PHASE1_MANUSCRIPT` §5.1**、**`FIGURE_CAPTIONS_STAGE1` P0** 与 **`metrics_result` basename** 一致；**`NEXT_RESEARCH_PLAN`** 收口清单更新 |
| 2026-04-10 | **§3 Phase M1**：**`SSGS_MAINLINE_M1.md`**；优先级表增 **M1**；**3090** 待办 **M1 优先**；**P1 B-S2+ CUDA** 标 **副线** |
| 2026-04-11 | **§3 M1**：harness **已落地** 叙事；里程碑改为 **P0 + M1 脚注** 并行；**七轴** 混读禁令；**P★** 与 **M1 L3** 分工 |
| 2026-04-11 | **§0**：全局阶段 / 分阶段路线图 / **下一步**；**七轴** + **SSGS 13 行** + **P1/P2 已划**；**pytest** 改为 **以实际计数为准** |
| 2026-04-11 | **§0.1**：**P0 收口 ≠ 研究终点**；指针 **`RESEARCH_STATUS` §1.5** 北星（快照回溯 × 树 RAG） |
| 2026-04-11 | **`RESEARCH_PHASES_0_TO_DONE.md`**：阶段 0–7 表 + 阶段 5 清单；篇首与 **`docs/README`** 互链 |
| 2026-04-11 | **§1 / 收口清单 / §B B3**：**AutoDL** 全量 **`python -m pytest tests/ -q`** — **28 passed**, **4 subtests**, **~21 s** |
| 2026-04-11 | **§3.1**：**M2** 后续实验 — **`SSGS_MAINLINE_M1.md` §6**（Wave A–D） |
| 2026-04-11 | **§3 / §3.1**：**§6.0** **B2 vs 全链条**；**§3** 完成句与 **`1247Z`** **n64 L3 CE** 归档一致 |
