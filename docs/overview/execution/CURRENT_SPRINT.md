# 当前迭代（滚动）

> **方向与现状总览**（读后再勾选）：**`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`**（含 **§3.5 对外叙事批判性接收、L1–L4 证据层级、P★ L3 PoC**）。  
> **全局阶段表**（**0→结题**、每阶段实验+成功标准）：**`docs/overview/planning/RESEARCH_PHASES_0_TO_DONE.md`**。**当前逻辑阶段**：**阶段 5 — 成文与投稿包**（阶段 0–4 已满足表内「完成」列）。  
> 每 1–2 周更新一次「周期」与勾选；完成后把结论一行写入 **`docs/experiments/planning/EXPERIMENT_REGISTRY.md`**。

## 周期

**开始**：2026-04-07  
**当前滚动至**：**2026-04-11** — **3090 批次已归档**（**`DATA_ARCHIVE_202604_SERVER.md`**）：**M1**、**SSGS**、**leavescale**、**headcheck**、**B-S2+ CUDA**、**L3 轨迹 JSON**。**执行焦点**：按 **`RESEARCH_PHASES_0_TO_DONE.md` §「阶段 5 检查清单」** 推进 **`SUBMISSION_PACK`** → 正稿；**P★ 训练型 L3** 仍 **非必做**（**`RESEARCH_STATUS` §3.5**）。

---

## 本周期焦点（**阶段 5 = P0 成文**）

| 优先级 | 动作 |
|--------|------|
| **1** | **阶段 5**：**`SUBMISSION_PACK.md` §A2** 路径与 **`EXPERIMENT_REGISTRY`** / **`DATA_ARCHIVE_202604_SERVER.md` §0** **逐字对齐** |
| **2** | **阶段 5**：**§A3** + **`FIGURE_CAPTIONS_STAGE1.md`** — **七轴** 脚注句 **粘贴进正稿**（或 **`PHASE1_MANUSCRIPT`** 对应节） |
| **3** | **阶段 5**：摘要/讨论/局限 与 **`RESEARCH_STATUS` §3.5** 一致（结论 **≤ L1–L3**） |
| **4** | ~~**3090 登记项**~~ **（已完成）**：**P1** B-S2+ CUDA、**P2** SSGS **n128** 等；见 **`DATA_ARCHIVE_202604_SERVER.md`**、**`NEXT_RESEARCH_PLAN` §2** |
| **5** | **（可选）** **`probe_retrieval_correlation.py --cpu`** / **§7 CPU** — 仅附录需要；**`NEXT_RESEARCH_PLAN` §B** |
| **6** | **`pytest tests/`**（或快测）提交前 **smoke** |

---

## 决策记录（真 LM 支线 → 主线）

**日期**：2026-04-09  

- **已完成**：**X-20260422–25**（最小 LM 闭环、启发式导航、goal 子头、叙事收束、**SSGS×LM** 并列；**CUDA** 指标与 `ssgs_lm_nav_compare_default8_cuda.json` 已登记 **EXPERIMENT_REGISTRY**）。  
- **定位**：该支线 **不替代** 主文 **path-batch 三 reader** harness；用于 **边界叙事** 与日后 **检索头 / 策略** 接口参考（见 `FIGURE_CAPTIONS_STAGE1.md` P0、`RESEARCH_NOTES` §7.0 / §7.4）。  
- **共识**：**优先回到主线** — **`docs/overview/planning/PROJECT_MASTER_PLAN.md` §1.1** 树内 **Mamba vs Transformer** 与阶段 1 **曲线/表** 的核查与成文；**B（MLP / 解冻 LM 抬 reach_rate）** **整体延后**，仅在单独排期或升格为正式子课题时再做。

---

## 本迭代目标（主线）

巩固阶段 1 **可交付素材**：**主图**（naive vs fused、与 `PHASE1_VALIDATION_PLAN` §6 一致）、**Wikitext 同 harness**、**§7 玩具协议**（S1–S4 + 复跑脚本）与 **登记册** 一一对应；避免将 **X-20260422–25** 数字与主图混读。

---

## 任务清单

- [x] 总体规划文档 `docs/overview/planning/PROJECT_MASTER_PLAN.md`
- [x] 扫参 CSV 增强：`gpu_name`、`torch_version`；合并多机 CSV 脚本 `scripts/benchmarks/merge_sweep_csv.py`
- [x] **文本形浅树**：样例叶文本 + 自底向上建树 + `scripts/benchmarks/benchmark_text_tree.py` + `run_reader_benchmark_on_paths`（确定性嵌入，非神经 encoder）
- [x] **数据约定**：`data/raw/sample/` 8 段合成 `.txt` + `docs/experiments/planning/DATASETS.md`；`scripts/data/prepare_leaves_from_corpus.py` 生成叶文件
- [x] **AutoDL 文档**：`docs/environment/runbooks/AUTODL_SETUP.md` + `SYNC` 索引；**已在 3090 实例跑通** smoke + `sweep_autodl.csv`（见 `EXPERIMENT_REGISTRY`）
- [x] **本地最小 Mamba**：`transformers.MambaModel` 小配置 smoke（无需 `mamba-ssm`），见 `scripts/smoke/smoke_mamba_minimal.py`
- [x] （可选）**mamba-ssm**：AutoDL 已装；**同机 naive 对照**见 `run_server_paper_main_sweep_naive.sh` + `SERVER_SWEEP_RUNBOOK` §2c（`mamba2_naive` 克隆环境卸融合栈）
- [x] **树路径三 reader**：`Mamba2PathReader` 接入 `benchmark_core` / `sweep` / `benchmark_text_tree`（默认开启，`--no-mamba2` 可关）
- [x] **公开语料浅树**：Wikitext-2（HF `datasets`）→ `hf_corpus.wikitext2_leaf_chunks` → `scripts/benchmarks/benchmark_wikitext_tree.py`（与合成叶同一 reader 槽位）
- [x] **3090 主文扫参（同机）**：`run_server_paper_main_sweep.sh`（fused）+ `run_server_paper_main_sweep_naive.sh`（`mamba2_naive`）；CSV 归档示例见本机 `results/metrics_result/`；图 `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`
- [x] **3090 Wikitext 浅树（fused + 镜像）**：`benchmark_wikitext_tree.py`；`results/metrics_result/benchmark_wikitext_3090_fused_20260408T0846Z.json`；登记 **A-20260408-wikitext-3090-fused**
- [x] **主文图注模板**：`docs/experiments/phases/FIGURE_CAPTIONS_STAGE1.md`
- [x] **§7.5 接线路线图** + **SSGS 固定 JSON**：`RESEARCH_NOTES.md`；**X-20260421-ssgs-tensor-overhead-fixed**；`benchmark_ssgs_tensor_overhead.py --out-json`
- [x] **3090 大叶数研究扫参**：`run_server_research_large_leaves.sh`，`TAG=research_lg_v1`；登记 **A-20260408-research-large-leaves-3090**
- [x] **S1 分段 cache 基准**：`benchmark_mamba2_cache_snapshot_segments.py`；**X-20260421-mamba2-cache-segments-cpu** + **X-20260421-mamba2-cache-segments-cuda**（`results/metrics/mamba2_cache_snap_segments_depth4_cuda_20260421.json`）
- [x] **S2 TF-R1（玩具路径，与 S1 对齐）**：`benchmark_tf_r1_path_segments.py`；**X-20260421-tf-r1-path-segments-cuda**（`results/metrics/tf_r1_path_segments_depth4_cuda_20260421.json`）
- [x] **S3 TF-KV**：`benchmark_tf_kv_path_segments.py`；**X-20260421-tf-kv-path-segments-cuda**（`tf_kv_path_segments_depth4_cuda_20260421.json` + **`tf_kv_path_segments_depth4_cuda_branchdemo_20260421.json`**）
- [x] **S4 SSM restore**：`benchmark_mamba2_cache_restore_segments.py`；**X-20260421-mamba2-cache-restore-cuda**（`mamba2_cache_restore_depth4_cuda_same_20260421.json` + **`mamba2_cache_restore_depth4_cuda_fromcpu_20260421.json`**）
- [x] **§7 协议复跑入口**：`scripts/research/run_path_protocol_cuda.sh` + `PHASE1_VALIDATION_PLAN` / `FIGURE_CAPTIONS` / `SERVER_SWEEP_RUNBOOK` 交叉引用
- [x] **叙事**：`FIGURE_CAPTIONS_STAGE1.md` 附录表注（中英）界定主图 vs §7.3.1
- [x] **P0 中文摘要**：主图 path-batch vs §7 玩具协议 vs SSGS 导航三线边界（`FIGURE_CAPTIONS_STAGE1.md` 篇首 + `RESEARCH_NOTES` §7.0）
- [x] **SSGS × Mamba**：`MambaNavState` / `dfs_ssgs_mamba` / `mamba_cache_utils.py`；`demo_ssgs_mamba_dfs.py`（`--out-json`）；`tests/test_ssgs_mamba.py`（默认网格 **events** 与登记 JSON 对齐）；登记 **X-20260421-ssgs-mamba-dfs-demo**
- [x] **真 LM 最小闭环（骨架）**：`src/rag_tree/tree_lm_closure.py` + `scripts/research/demo_tree_lm_minimal.py`（路径文本 → CE + 续写 + 可选一步训练；`--out-json` 写 **`git_sha`**）；登记 **X-20260422-tree-lm-minimal**
- [x] **树上导航任务指标（启发式）**：`tree_lm_nav_eval.py` + `demo_tree_lm_nav_greedy.py`（子文档 CE argmin、**reach_rate** / **child_choice_accuracy**）；登记 **X-20260423-tree-lm-nav-greedy**
- [x] **目标叶条件可学习子指针**：`tree_lm_nav_learned.py` + `demo_tree_lm_nav_learned.py`（冻结 LM、goal 嵌入 + 线性头；**X-20260424**；**CPU/CUDA** 归档 `tree_lm_nav_learned_default8_{cpu,cuda}.json`，**reach_rate=0.375** 一致）
- [x] **叙事收束（C）**：`FIGURE_CAPTIONS_STAGE1.md` P0 段末 + `RESEARCH_NOTES` §7.0 / §7.4 — **X-20260424** 默认 **reach_rate** 仍低于 1，勿过度宣称；与 SSGS **任务不同**
- [x] **SSGS×LM 并列（A）**：`demo_ssgs_lm_nav_compare.py`（同文本 8 叶：**dfs_ssgs_mamba** 必达 vs **子头贪心**）；`tests/test_ssgs_lm_nav_compare.py`；登记 **X-20260425**；**CUDA** 归档 `ssgs_lm_nav_compare_default8_cuda.json`

### 主线待办（本周期优先）

- [x] **主图与登记对齐审计**：见 **`PHASE1_VALIDATION_PLAN.md` §6.5**（2026-04-09）：**A-20260408-paper-main-3090-*** ↔ `results/metrics_result/paper_main_*_{v1,naive_v1}.csv` ↔ `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`。  
- [x] **`PHASE1_VALIDATION_PLAN.md` 结论段**：**§6.3**（结论文本）+ **§6.5**（引用规则）；**5060 vs 3090** 不可混填见 **§6.2**。  
- [x] **§7 可复现性**：**2026-04-08** AutoDL 上已按附录 A 全量跑通 `run_path_protocol_cuda.sh`（`MAMBA2_RESULTS_ROOT` → `…/mamba2_results/metrics/`，`STAMP=20260408T1617Z`）；数值与 **`RESEARCH_NOTES` §7.3.1** / 仓内 `*_20260421.json` **同阶**，见 **`PHASE1_COMPLETE_SUMMARY.md` 附录 B**。  
- [x] **阶段 2 里程碑表**：**`NEXT_RESEARCH_PLAN.md` §2**（**`ROADMAP.md`** 仅入口指针）。

### 后续两周执行计划（主线，可勾选）

- [x] **成文**：**`docs/experiments/phases/PHASE1_MANUSCRIPT.md`**（摘要—方法—结果—§7 关系—**`results/metrics_result/`** 索引—结论文本—英文摘要）；正文可整节迁移或按需截取 **§6** 与 **`FIGURE_CAPTIONS_STAGE1.md`** 句稿。  
- [x] **指标归档**：主文 CSV、Wikitext JSON、§7 复跑 `*_20260408T1617Z.json`、大叶数扫参等已集中于 **`results/metrics_result/`**（本机 `D:\cursor_try\mamba2\results\metrics_result`），并已纳入 **Git** 与 **§6.5** 表。  
- [x] **阶段 2 开工**：按 **`NEXT_RESEARCH_PLAN.md` §2** — **A2-S0** 登记占位 + **A2-S1** `benchmark_wikitext_tree.py` smoke（**`--out-json`** 至 **`results/metrics_result/`**，见该文档 **§2.4**）；**B-S1** **`docs/research/RETRIEVAL_HEAD_NOTES.md`**。  
- [x] **检索头 B-S2（本地可完成）**：**`probe_retrieval_correlation.py`** + **`RETRIEVAL_HEAD_NOTES.md` §2** 文献入口表（与 2404.15574 等对照）；**per-head / 大模型** 仍待 **B-S3** 与 GPU。  
- [x] **B-S2+ path reader 探针**：**`probe_path_reader_linear.py`**（默认 **16 叶 heldout**、**`ridge_untrained`**；可选 **`bce_reader_train` / `bce_head_only_train`**）；归档 **`results/metrics/probe_path_reader_linear_text16_*.json`**。  
- [x] **A2-S3 任务 smoke 扩展**：**`task_wikitext_path_pair.py`** — **`task_wikitext_path_pair_sibling16_cpu.json`**、**`task_wikitext_path_pair_rootchild16_cpu.json`**（**CPU**，`split_seed=1`）；总览 **RESEARCH_STATUS** §3 测量轴、**`PHASE2_DRAFT.md`** §2。  
- [x] **A2-S3 叶级 heldout**：**`--pair-split leaf_heldout --heldout-leaves 4`**（16 叶 sibling）→ **`task_wikitext_sibling16_leafheldout4_{cpu,cuda5060}.json`**；见 **`PHASE2_DRAFT.md`**（小 **test** 叶对数、CPU/CUDA ridge 波动）。  
- [x] **本地收尾**：**`path_pair_geometry.py`** + **`tests/test_path_pair_geometry.py`**；**`aggregate_wikitext_5060_cuda_grid.py`** → **`benchmark_wikitext_5060_cuda_grid_20260407.csv`**；**A2-S3** **`chunk_len=12`** **`…_c12_leafheldout6_*.json`**；**`PHASE1_MANUSCRIPT` §3.1**。  
- [x] **P1 成文并入主稿**：**`PHASE1_MANUSCRIPT.md` §8–§10**（阶段 2 系统+A2-S3、检索头/Mamba 边界、文档指针）；**`FIGURE_CAPTIONS_STAGE1.md`** **七条测量轴**表（含 **M1**、**L3 轨迹**）；**`PHASE2_DRAFT.md`** 顶注指向主稿。  
- [x] **仓库梳理（代码/文档）**：**`src/rag_tree/__init__.py`** 惰性导出（**`test_path_pair_geometry`** 可不加载 **torch**）；根目录 **`pytest.ini`**；**`PHASE1_MANUSCRIPT` §5** 主图路径改为 **`results/metrics/figures/`**；**`scripts/README.md`** 补 **aggregate** / **task_wikitext**；**`docs/README.md`** 补 **PHASE2_DRAFT** 链；根 **`README.md`** 单测分层说明。  
- [ ] **脚本卫生**：Linux 上若再遇 **`bash\r`**，对 **`scripts/**/*.sh`** 执行 **`find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'`**（见 **`SH_CRLF_LINUX.md`**）。
- [x] **A2-S2 第二轮（3090 fused）**：**`TAG=stage2_fused_r2`** **`STAMP=20260409T1110Z`**；**Mamba2 峰值与 R1 一致**；**`git_sha` JSON 内仍为 `6fa7873`**；归档 **`metrics_result/benchmark_wikitext_stage2_fused_r2_*`**；登记 **A-stage2** 已更新。

### 文档与代码检查纪要（2026-04-09）

| 项 | 结论 |
|----|------|
| **docs** | `EXPERIMENT_REGISTRY` / `PHASE1_*` / `FIGURE_CAPTIONS` / `RESEARCH_NOTES` §7 与 **主线辅线** 分工一致；未发现 `TODO/FIXME` 占位。 |
| **`run_path_protocol_cuda.sh`** | 已 **LF** + **`set -eu` / `set -o pipefail` 分行**；**`.gitattributes`** 已约束 `*.sh eol=lf`。 |
| **其它 `.sh`** | 共 7 个；若从 Windows 上传覆盖，仍可能 CRLF — 用 **`SH_CRLF_LINUX.md`** 批量 `sed`。 |
| **§7 基准脚本** | `benchmark_*_path_segments.py` 等向 **stdout 打 JSON** 属预期；与 **`--out-json`** 写入文件并行。 |

### 文档与代码检查纪要（2026-04-07，续）

| 项 | 结论 |
|----|------|
| **路径** | **`PHASE1_MANUSCRIPT` §5** 主图路径修正为相对仓库根 **`results/metrics/figures/`**（原 `../metrics/…` 从 `docs/experiments/phases/` 会误解析）。 |
| **导入** | **`src.rag_tree`** 包级不再在 **`__init__`** 中强依赖 **`torch`**；**`tests/test_path_pair_geometry.py`** 可在无可用 PyTorch DLL 的环境完成收集与通过。 |
| **索引** | **`scripts/README`**、**`docs/README`**、**`RESEARCH_STATUS`** 与 **P1 成文**、**A2-S2 待办** 对齐。 |

### 支线（延后，非本周期默认）

- [ ] **B**：抬高 **X-20260424** 子头（MLP、`goal_dim`、epoch/lr、小步解冻 LM）；**X-20260425** 加 **wall-clock** — 仅在 **GPU 空闲** 或 **导航升格为正式贡献** 时启动。

---

## 阶段 5 检查清单（与 **`RESEARCH_PHASES_0_TO_DONE.md`** 同步）

- [ ] **§A2** basename ↔ 登记册  
- [x] **§A3 / §A3b** 七轴脚注：**中文 §A3** + **英文 §A3b** 已在 **`SUBMISSION_PACK.md`** — 粘贴进 LaTeX/Word  
- [ ] **局限段** 含 §3.5 三风险（有损状态 / 回退触发 / H2D）— **`PHASE1_MANUSCRIPT` §9.2** 已有骨架  
- [x] **`git push`**：**`7052efa`** 已上 **`origin/master`**（含 metrics + 文档 + L3 轨迹 harness）  
- [ ] **`pytest tests/`**（或聚合单测；须 **torch** 环境）  

---

## 阻塞项

- **AutoDL / 3090**：**实例当前可用**（**2026-04**）；**原 P1/P2** 已归档。**新跑次** 按 **`NEXT_EXPERIMENTS_COMMANDS.md`**；若断档，改 **「算力不可用时的备选推进」**（**`NEXT_RESEARCH_PLAN`**）并 **更新本行日期**。
- **云端算力（常设）**：**按需开机**；主环境 **`conda activate mamba2`** 下 fused 已验证。**检索头训练**等仍受机时与预算约束。
- **仅本机 5060 时**：无法复现 3090 fused 数字；以登记册与图为准，不混填表格。

---

## 优先事项（本机 + 云端）

| 优先级 | 方向 | 可执行项 |
|--------|------|----------|
| P0 | **主线：阶段 1 成文素材** | 主图/CSV 与 **`docs/experiments/planning/EXPERIMENT_REGISTRY.md`** 对齐；**`docs/experiments/planning/PHASE1_VALIDATION_PLAN.md`** 结论段定稿；**`docs/experiments/phases/FIGURE_CAPTIONS_STAGE1.md`** / **§7.0** 作 **口径护栏**；**`docs/experiments/phases/PHASE1_MANUSCRIPT.md`** 已含 **阶段 2 §8–§10** 与 **七轴**（含 **M1** **`ssgs_vs_kv_wikitext_nav_grid.csv`**）交叉引用；**`SUBMISSION_PACK.md` §A3–A2.1** **粘贴进正稿**（截稿前仍可润色） |
| P0 | **真实语料线（云端）** | （已完成）3090 + `HF_ENDPOINT`；**A-20260408-wikitext-3090-fused** — 主线引用时标明 **与合成树同一 harness** |
| P1 | **主线：§7 协议** | **复跑已通过**（见 sprint §7 勾选 + **`PHASE1_COMPLETE_SUMMARY` 附录 B**）；正文仍须与主图 **分列声明**（§7.3.1） |
| P1 | **SSGS（协议层）** | **X-20260421-*** 张量 + **`dfs_ssgs_mamba`** demo；**不等于** 真 LM 导航线 |
| P2 | **阶段 2 执行** | **`NEXT_RESEARCH_PLAN.md`** 轨道 **A**：**A2-S3** + **5060 CUDA** Wikitext **`n8_c12` / `n16_c12`**（**`metrics_result/benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json`** + **`PHASE2_DRAFT` §1.1**）；**A2-S2 fused** 已有归档，**新格点** 在 **3090 可用时** 按 **`SERVER_SWEEP_RUNBOOK`** 续跑 |
| P2 | **检索头 B** | 探针与层/头报告（与主线并行，需 **48G** 窗口） |
| 延后 | **真 LM 支线 B** | **X-20260424** 子头加强、**X-20260425** wall-clock；**非**阶段 1 unblock |
| 延后 | **大叶数扩展** | **A-20260408-research-large-leaves-3090** 已归档；新网格另开 **TAG** |

**讨论结论（写入此表的目的）**：主故事仍是 **`docs/overview/planning/PROJECT_MASTER_PLAN.md` §1.1 树内 Mamba vs Transformer**；**X-20260422–25** 为 **辅线登记**，回归主线后 **默认不增投** 直至阶段 2/检索头需要接口演示。

---

## 后续研究方向（滚动，与总览一致）

1. **近期（阶段 1 收尾）**：主图 + §7 **双轨叙事**定稿；registry / 图 / CSV **三角互证**；5060 本地只跑 **smoke 与结构校验**，**3090 数字以登记为准**。  
2. **中期（阶段 2）**：在 **真语料** 上建 **浅层树**（Wikitext 扩展或 RAPTOR 式），**同一** `benchmark_*_tree` / reader 接口，补 **1 个任务指标**（导航或 QA）。  
3. **并行（检索头）**：`PROJECT_MASTER_PLAN` **B→C**；与 **A** 共享小树探针，再上 **48G** 注入实验。  
4. **长期（协议贡献）**：**SSM 快照 vs TF-KV** 在 **同等试错轨迹** 下的对照表（§7.5 S5）；真 LM 导航仅在有 **新假设** 时再接回。

---

## 上迭代归档（简述）

- 环境 `mamba2`、cu128、玩具树基准、本地 preset 扫参 8 点 CSV。
- **2026-04-08**：3090 `paper_main_v1` / `paper_main_naive_v1` 成对数据与同机三张 `mamba_3090_naive_vs_fused_*.png`（见 `EXPERIMENT_REGISTRY`）。
- **2026-04-11**：**`DATA_ARCHIVE_202604_SERVER.md`** 入仓；**M1 / SSGS / leavescale / B-S2+ CUDA** 路径与 **CSV `json_path`（相对仓）** 对齐；**聚合脚本** 在 JSON 位于 **`cwd`** 下时写 **POSIX 相对路径**。
- **2026-04-11**：**阶段 C** **L3 轨迹** **`tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`** 入 **`metrics_result/`**；登记 **X-20260411-tf-kv-trajectory-l3-minimal** 补数值行。
