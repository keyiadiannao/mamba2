# SSGS 整合主线 · Phase M1（正式开工）

> **定位**：在 **Mamba + 树 + SSGS** 主叙事下，从「各轴可行」推进到 **L3 级对照**（见 **`RESEARCH_STATUS_AND_DIRECTION.md` §3.5**）：**同一树任务**上 **隐状态快照回溯** 相对 **KV/重算类基线** 的 **代价 + 行为** 可发表对比。  
> **检索探针（B-S2/B-S2+）** 仍为 **副线**，不纳入本页里程碑。  
> **权威排序**仍服从 **`NEXT_RESEARCH_PLAN.md`**；本页为 **M1 专用工具清单与检查表**。

---

## 1. 工具就绪检查（已有）

| 能力 | 代码入口 | 脚本 / 模块 | 测试 |
|------|----------|-------------|------|
| **SSGS 核心（离散 + 张量 + Mamba）** | `src/rag_tree/ssgs.py` | `dfs_ssgs`、`dfs_ssgs_tensor`、`dfs_ssgs_mamba`、`MambaNavState`、`SSGSTrace` | `tests/test_ssgs.py`、`tests/test_ssgs_mamba.py` |
| **Mamba cache clone / restore** | `src/rag_tree/mamba_cache_utils.py` | 与 §7 S1/S4 同语义 | §7 基准脚本见下 |
| **SSGS × 玩具树 × Mamba** | — | `scripts/research/demo_ssgs_mamba_dfs.py` | `tests/test_ssgs_mamba.py`（间接） |
| **SSGS × Wikitext 同建树** | 与 `benchmark_wikitext_tree` 同 `wikitext2_leaf_chunks` → `build_bottom_up_text_tree` | `scripts/research/demo_ssgs_mamba_wikitext.py` | `tests/test_ssgs_mamba_wikitext.py` |
| **SSGS 结果汇总** | — | `scripts/research/aggregate_ssgs_mamba_wikitext_json.py` | `tests/test_aggregate_ssgs_mamba_wikitext_json.py` |
| **云端一键（CUDA + smoke 可选）** | — | `scripts/server/bootstrap_autodl.sh`、`scripts/server/run_ssgs_mamba_wikitext_cuda.sh` | — |
| **M1 云端：叶数扫（三臂）** | — | **`scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**（**`M1_LEAVES`** / **`M1_NO_TRUNCATE`**） | — |
| **SSGS × LM 并列（辅）** | 同文本 8 叶 | `scripts/research/demo_ssgs_lm_nav_compare.py` | `tests/test_ssgs_lm_nav_compare.py` |
| **§7：Mamba 段快照/恢复 ms** | 单路径 | `scripts/research/benchmark_mamba2_cache_snapshot_segments.py`、`benchmark_mamba2_cache_restore_segments.py` | — |
| **§7：TF-R1 / TF-KV（含错枝截断 demo）** | 单路径玩具 trunk；**KV 类** 亦在 `src/rag_tree/tf_kv_incremental.py` | `scripts/research/benchmark_tf_r1_path_segments.py`、`benchmark_tf_kv_path_segments.py`（`--branch-truncate-demo`） | — |
| **M1：SSGS vs TF-KV 同树 DFS（Wikitext）** | 与 `demo_ssgs_mamba_wikitext` 同建树 / `target_leaf_index`；可选 **`--l3-tf-kv-hidden`** | `scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py` | `tests/test_tf_kv_tree_nav.py`、`tests/test_tf_kv_l3_probe.py` |
| **M1：多 JSON → 网格 CSV** | **`kind=ssgs_vs_kv_tree_nav_wikitext`**；可选 **L3** 列 | `scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py` | `tests/test_aggregate_ssgs_vs_kv_wikitext_json.py` |
| **M1：L3 隐状态一致性（TF-KV）** | DFS+restore 末 token hidden vs 金路径纯增量前向（同权重） | `src/rag_tree/tf_kv_l3_probe.py` | `tests/test_tf_kv_l3_probe.py` |
| **M1：L3 下游 CE（固定叶头）** | 未训练 **`Linear(dim,num_leaves)`** 在 nav / 金路径 hidden 上的 CE；**`abs_ce_delta≈0`** 校验对齐。与 **X-20260423/24** 树 LM「CE 路由 vs **可学习**子头」**不同 harness**（后者可更强，仅叙事参考） | `src/rag_tree/tf_kv_l3_downstream_probe.py` | `tests/test_tf_kv_l3_downstream_probe.py` |
| **path-batch 同树三 reader** | Wikitext | `scripts/benchmarks/benchmark_wikitext_tree.py` | — |
| **path-batch + SSGS 同树 smoke（三 reader 一行 JSON）** | — | `run_ssgs_mamba_wikitext_cuda.sh` 中 **`RUN_WIKITEXT_SMOKE=1`** → `benchmark_wikitext_tree` bundle | — |
| **SSGS 张量微基准（无 LM）** | — | `scripts/benchmarks/benchmark_ssgs_tensor_overhead.py` | — |

**结论（工具层）**：**L1（机制可行）** 与 **Wikitext 上 SSGS 全流程** **齐全**；**§7 玩具协议** 提供 **KV 截断 / 增量** 叙事与 **毫秒/nbytes** 数字。

---

## 2. 缺口（M1 待补）

| 缺口 | 说明 | 建议产出 |
|------|------|----------|
| **统一 harness「同树 · 同任务 · 多臂」** | **已有**：`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`（``kind=ssgs_vs_kv_tree_nav_wikitext``）：**Mamba** + **TF-KV clone** + **TF-KV truncate**；**``--l3-tf-kv-hidden``** → **隐状态 L3**；``--no-tf-kv-truncate`` 可关第三臂；**``tf_kv_arm``** = **``tf_kv_clone_arm``**（兼容） | **已登记** **`X-ssgs-vs-kv-tree-nav-m1`**；**叶扫** **``20260410T1012Z``** |
| **L3 语义对照** | **隐状态（已有）**：**`--l3-tf-kv-hidden`**。**下游（最小 CE）**：**`--l3-tf-kv-downstream-ce`** → **`l3_tf_kv_downstream_ce`**（**固定随机**叶分类头，**非**训练树 LM 子头）。**树 LM 叙事**：**X-20260424** 可学习导航可优于 **X-20260423** 的 CE argmin —— **不同任务**，此处仅脚注参考 | **隐状态**：**`tf_kv_l3_probe`**；**下游 CE**：**`tf_kv_l3_downstream_probe`**；**禁止**与 path-batch 主表无脚注合并 |
| **树上 TF-KV 与 path-batch 的公平脚注** | `TransformerPathReader` 为整段 SA；§7 TF-KV 为 **玩具 trunk** —— M1 须在 JSON/README 中 **写清基线定义** | 登记册 **「与何物对照」** 一栏一段话 |

---

## 2.1 实测归档摘要（**`X-ssgs-vs-kv-tree-nav-m1`**）

> **登记真相**与脚注：**`docs/experiments/planning/EXPERIMENT_REGISTRY.md`** 表内 **`X-ssgs-vs-kv-tree-nav-m1`**。下列为 **`STAMP=20260410T1012Z`**、**三臂**、**`wikitext-2-raw-v1`**、**f2 / c8 / dim128**、**目标最右叶** 的 **`wall_s`（秒）** 量级（**跨臂不对等**，勿单独比快慢叙事）。

| **`num_leaves`** | **迹** `snapshots` / `rollbacks` / `leaf_checks` | **Mamba** `wall_s` | **TF-KV clone** `wall_s` | **TF-KV truncate** `wall_s` | **`truncate_kv_calls`** | **`kv_nbytes_at_end`** |
|------------------|---------------------------------------------------|--------------------|---------------------------|-----------------------------|-------------------------|-------------------------|
| 8 | 7 / 11 / 8 | ≈0.546 | ≈0.081 | ≈0.038 | 14 | 65536 |
| 16 | 15 / 26 / 16 | ≈0.783 | ≈0.118 | ≈0.069 | 30 | 81920 |
| 32 | 31 / 57 / 32 | ≈1.21 | ≈0.185 | ≈0.132 | 62 | 98304 |
| 64 | 63 / 120 / 64 | ≈2.03 | ≈0.31 | ≈0.26 | 126 | 114688 |

**文件**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n{8,16,32}_cuda_3arm_20260410T1012Z.json`**；**n64（无 L3）** **`…_n64_cuda_3arm_20260410T1235Z.json`**；**n64 复跑**：**`…_n64_cuda_3arm_20260410T1616Z.json`**（三臂、**无** L3 CE 块）；**`…_n64_cuda_3arm_20260410T1617Z.json`**（三臂 **+** **`l3_tf_kv_downstream_ce`**，**`abs_ce_delta`=0**，**`ce`≈4.29/4.16**）。另：**首轮无 STAMP** 的 **`…_n8_cuda_3arm.json`** 可作对照。

**趋势（一句）**：**Mamba** `peak_alloc_mib` 三档均 **≈130**；**TF-KV** **≈27.7–28.1**；**`wall_s`** 随叶数上升主要来自 **DFS 迹长度**，**truncate** 臂略快于 **clone** 臂（同 harness）。

**L3 隐状态（n8 CUDA）**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm_l3.json`** — **`l3_tf_kv_hidden`**：**clone / truncate** 与金路径-only 的 **末 token hidden** **余弦 ≈ 1**、**`l2_diff`=0**（**`git_sha` 以 JSON 为准**）。**复跑（`STAMP=20260410T1244Z`）**：**`…_n8_cuda_3arm_20260410T1244Z.json`** — 同结构 **cosine ≈ 1**、**`l2_diff`=0**。

**L3 下游 CE（n8 CUDA，`STAMP=20260410T1113Z`）**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm_20260410T1113Z.json`** — **`l3_tf_kv_downstream_ce`**（**`--l3-tf-kv-downstream-ce`**）：**clone** **`ce_nav`=`ce_ref`≈2.121**、**truncate** **`ce_nav`=`ce_ref`≈2.224**、两臂 **`abs_ce_delta`=0**、**`max_abs_logit_diff`=0**（**`probe_seed`=12345**）。同次跑 **Mamba** **`wall_s`≈0.507**、**clone**≈0.074、**truncate**≈0.031（与 **1012Z** 同阶，GPU 抖动正常）。

**L3 下游 CE（n16 / n32 CUDA，`STAMP=20260410T1133Z`）**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n16_cuda_3arm_20260410T1133Z.json`**、**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n32_cuda_3arm_20260410T1133Z.json`** — **`l3_tf_kv_downstream_ce`**：**n16** **clone** **`ce_nav`=`ce_ref`≈2.645**、**truncate** **≈2.708**；**n32** **clone** **≈3.430**、**truncate** **≈3.489**；各臂 **`abs_ce_delta`=0**、**`max_abs_logit_diff`=0**。同次三臂 **`wall_s`** 与 **1012Z** 同档（**n16** **Mamba**≈0.774、**n32**≈1.22，GPU 抖动）。

**L3 下游 CE（n8–n64 同批 CUDA，`STAMP=20260410T1247Z`）**：**`…_n{8,16,32,64}_cuda_3arm_20260410T1247Z.json`** — 各叶数两臂 **`abs_ce_delta`=0**、**`max_abs_logit_diff`=0**（例 **n64** **clone** **`ce_nav`=`ce_ref`≈4.177**、**truncate** **≈4.252**，**`probe_seed`=12345**）。

### 2.2 **阶段 C：轨迹甲·乙**（**≠ M1 全 DFS**）

**`RESEARCH_STATUS` §3.5** 最小对照：**硬编码** 读序 — **轨迹乙** 仅金路径；**轨迹甲** 根 → **快照** → **错子一步** → **restore** → 金路径后缀。**`kind=tf_kv_trajectory_l3_minimal`**。实现：**`src/rag_tree/tf_kv_trajectory_l3.py`**、**`scripts/research/benchmark_tf_kv_trajectory_l3_minimal.py`**；登记 **X-20260411-tf-kv-trajectory-l3-minimal**。**禁止**与 **M1**、**path-batch** 无脚注同表。

**已归档（3090 CUDA，`STAMP=20260410T1341Z`）**：**`results/metrics_result/tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`** — **clone** **`cosine_hidden_a_vs_b`≈0.99999988**、**truncate_kv** **≈0.99999994**，两臂 **`l2_diff_hidden_a_vs_b`=0**（**`git_sha=6fa7873`**）。

**网格 CSV**：**`results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`** — **`python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py -g 'results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json' --out-csv results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`**。云端 **`run_m1_ssgs_vs_kv_wikitext_cuda.sh`** 扫叶后默认聚合（**`SKIP_M1_AGGREGATE=1`** 跳过；**`AGGREGATE_APPEND=1`** 追加）。

---

## 3. Phase M1 目标（4 周内可检查）

1. **选定基线臂**：优先 **TF-KV 玩具 trunk 上的「错枝 + 截断 KV + 兄弟子树」**（与现有 **`--branch-truncate-demo`** 语义衔接），或 **「回到分叉点整段重算」** 二选一写死。  
2. **选定任务**：与现有 **SSGS** 一致的最右叶 / 或 **goal 叶**（与 **`demo_ssgs_mamba_wikitext`** 的 **`target_leaf_index`** 对齐）。  
3. **统一报告**：**总前向步数或 token 数**、**wall-clock**、**峰值显存**、**snapshots/rollbacks**（SSGS 臂）、**kv 字节或截断次数**（KV 臂）、**任务成功**（`ok`）。  
4. **登记**：**`EXPERIMENT_REGISTRY.md` 新行**；JSON 入 **`results/metrics_result/`**；**`FIGURE_CAPTIONS_STAGE1.md`** 若新增对照图须 **显式测量轴脚注**（当前 **七轴** 含 **L3 轨迹**）。

---

## 4. 开工检查表（复制到 issue / sprint）

- [x] 读完 **`RESEARCH_STATUS_AND_DIRECTION.md` §3.5**（证据层级与风险三段）— **成文前必读**；局限成句见 **`PHASE1_MANUSCRIPT` §9.2**。  
- [x] **单测**：**AutoDL** **`python -m pytest tests/ -q`** — **28 passed**, **4 subtests**（**2026-04-11**）；**`test_ssgs*.py`** / **`test_tf_kv_*`** 须 **可用 torch** 的环境。**无 torch**：**`py -3 -m pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py tests/test_aggregate_ssgs_vs_kv_wikitext_json.py -q`**。  
- [x] 在 **`EXPERIMENT_REGISTRY`** 登记 **M1**：**`X-ssgs-vs-kv-tree-nav-m1`** + **`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm.json`**（三臂，**`git_sha=6fa7873`**）；历史两臂文件 **`…_n8_cuda.json`** 仍保留作对照。  
- [x] **叶数扩展**：**`STAMP=20260410T1012Z`** — **`…_n{8,16,32}_cuda_3arm_20260410T1012Z.json`**；**n64** **`STAMP=20260410T1235Z`** — **`…_n64_cuda_3arm_20260410T1235Z.json`**；已写入 **`EXPERIMENT_REGISTRY`** **X-ssgs-vs-kv-tree-nav-m1**。  
- [x] **L3（隐状态）**：**`--l3-tf-kv-hidden`** + **`src/rag_tree/tf_kv_l3_probe.py`**；单测 **`tests/test_tf_kv_l3_probe.py`**。  
- [x] **网格 CSV**：**`aggregate_ssgs_vs_kv_wikitext_json.py`** → **`ssgs_vs_kv_wikitext_nav_grid.csv`**；单测 **`tests/test_aggregate_ssgs_vs_kv_wikitext_json.py`**。  
- [x] **L3（下游 CE，固定叶头）**：**`--l3-tf-kv-downstream-ce`**；单测 **`tests/test_tf_kv_l3_downstream_probe.py`**。树 LM **可学习 vs CE**（**X-20260423/24**）为**另一 harness**，不作数值可比。  
- [x] **L3 下游 CE · n16/n32（CUDA）**：**`STAMP=20260410T1133Z`** — **`…_n{16,32}_cuda_3arm_20260410T1133Z.json`**；**`abs_ce_delta`=0**；已写入 **登记册** / §2.1。  
- [x] **L3（轨迹甲·乙，玩具 TF-KV）**：**`tf_kv_trajectory_l3_minimal`** + **`tests/test_tf_kv_trajectory_l3.py`**；**JSON** 按需入 **`metrics_result/`**。  
- [ ] **L3（训练型子头 / 与树 LM 对齐）**：若要做，须另 **`kind`** 与登记；**禁止**与 path-batch 主表无脚注合并。

---

## 5. 相关文档

- **`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`** — 七轴、§3.5、L3 最小验证。  
- **`docs/overview/execution/NEXT_RESEARCH_PLAN.md`** — **M1** 与 **P0 成文 / P1 B-S2+ CUDA** 并行关系。  
- **`docs/research/RESEARCH_NOTES.md`** §6–§7 — SSGS 与 §7 边界。  
- **`scripts/README.md`** — 脚本索引（含 SSGS / §7）。  
- **M1 云端扫叶数**：**`scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**（默认 **`M1_LEAVES="8 16 32"`**、**三臂**；**`M1_NO_TRUNCATE=1`** 仅两臂）。

---

## 6. 后续实验跑道（**M2**：M1 归档之后）

> **定位**：**M1** 三臂 + 叶扫 + **L3**（隐状态 / 下游 CE）+ **轨迹 minimal** 已满足 **L2–L3** 叙事；**M2** 不再重复「同一网格再扫一遍」，除非 **审稿点名** 或 **代码版本（`git_sha`）** 需刷新。  
> **原则**：新 JSON **新 `STAMP`**、**登记册新行或脚注**；与 **path-batch**、**§7 毫秒**、**A2-S3** **脚注分列**（**七轴**）。

### 6.0 **B2 在做什么？与「全链条」的关系**（上 AutoDL 前读）

本仓 **「Mamba + 树状 RAG + SSGS」** 在 **M1 harness**（**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`**）里已串好的 **可实现链条** 是分层来的，**不是**「一句端到端大模型 RAG」：

| 环节 | 内容 | **B2 是否覆盖** |
|------|------|-----------------|
| **① 数据 → 浅树** | **Wikitext-2** 叶块 → **`build_bottom_up_text_tree`**（与 **`demo_ssgs_mamba_wikitext`** 同协议） | ✅ 每次 M1 跑都会走 |
| **② 同树 DFS 导航（三臂）** | **SSGS+Mamba**；**TF-KV clone**；**TF-KV truncate**；记 **`wall_s`、峰值、`ok`、快照/回滚** 等 | ✅ **B2 命令里必然执行**（与是否开 L3 无关） |
| **③ L3 探针（可选）** | **隐状态一致**（`--l3-tf-kv-hidden`）或 **固定随机叶头上 CE 对齐**（`--l3-tf-kv-downstream-ce`）：检验 TF-KV 臂上「走错再恢复」的轨迹与「只走金路径」在表示上是否一致 | **B2 只做其中一种**：把 **③ 的下游 CE** 从已有 **n8/n16/n32** **补到 n64**，使 **网格 CSV** 里 **L3 CE 列** 与 **叶数** 对齐，**不**引入新对比算法 |
| **④ 其它对比方法** | 例如新回溯基线、可学习策略、更大 LM —— 须 **新臂 / 新 `kind` + 登记**（**§6.4** / **`PROJECT_MASTER_PLAN` §1.0**） | ❌ **B2 不包含**；应在 **①②（及可选③）跑通** 后再单独立项 |

**结论**：**B2 不是「从零验证全链条」**，而是 **在同一条 M1 harness 上** 跑 **n64** 且 **打开 L3 下游 CE**（**环节 ②+③**）。**本仓已归档** 同批 **`STAMP=20260410T1247Z`** 的 **n8/n16/n32/n64** 四 JSON（均含 **`l3_tf_kv_downstream_ce`**，**`abs_ce_delta`≈0**）— 若你 **AutoDL 工作区已有这些文件**，**B2** 的目的改为 **换机复现 / `git pull` 后刷新 `git_sha`**；**若缺失 n64 该列**，再用 B2 **补档**。若你要 **先验证全链条可实现、再考虑加新方法**，建议在 AutoDL 上 **分两步**：

1. **全链条 smoke（可不开 L3，先确认 ② 三臂在 n64 上 `ok`）**  
   `M1_LEAVES="64" bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`  
   （不设 **`M1_WITH_L3_*`**，JSON 仍含 **mamba / tf_kv_clone / tf_kv_truncate** 三臂与 **`ok`**。）
2. **再上 B2（在 ①② 已确认的前提下，补 L3 CE 列）**  
   `M1_LEAVES="64" M1_WITH_L3_DOWNSTREAM_CE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`  
   核对 JSON 内 **`l3_tf_kv_downstream_ce`**、各臂 **`abs_ce_delta`** 应与 **n8/n16/n32** 归档同阶（≈0）。

**与 path-batch 主图的关系**：**`benchmark_wikitext_tree`**（多路径 batch 前向）是 **另一条测量轴**；「全链条」若还包括 **同树 path-batch + SSGS 一行 JSON**，见 **`run_ssgs_mamba_wikitext_cuda.sh`** 里 **`RUN_WIKITEXT_SMOKE=1`**（**`NEXT_EXPERIMENTS_COMMANDS.md` §10**），与 **M1 DFS** **脚注分列**。

**再加「其它方法」做对比**：在 **①②** 稳定后，为 **新基线** 单开实验设计与 **`EXPERIMENT_REGISTRY`** 行；**禁止**与当前 M1 表无脚注合并（**七轴**）。

### 6.1 Wave **A** — 与成文并行（**不占 GPU**）

| 顺序 | 动作 | 产出 |
|------|------|------|
| A1 | **`SUBMISSION_PACK` §A3 / §A3b** + **§A2.1** 入 LaTeX/Word | 主稿脚注与 **M1** **`ssgs_vs_kv_wikitext_nav_grid.csv`** 一句 |
| A2 | **`FIGURE_CAPTIONS_STAGE1.md`** 七轴表 ↔ 正文 **逐轴** 出现或明确「见附录」 | 防混读审稿 |
| A3 | （可选）**`RESEARCH_NOTES` §7.5 S5** 总表 | 仅篇幅允许时 |

### 6.2 Wave **B** — 云端 **单条增量**（**AutoDL**；每条 **≤ 数分钟级**，按需选 **1–2** 条）

**环境**：**`docs/environment/runbooks/AUTODL_SETUP.md`** + **`scripts/server/_autodl_env.sh`**（**`conda activate mamba2`**、**`MAMBA2_RESULTS_ROOT`**、**`HF_ENDPOINT`**）。

| 优先级 | 实验 | 命令要点 | 登记 / 后处理 |
|--------|------|----------|----------------|
| **B1** | **`git_sha` 刷新**（代码更新后） | **`M1_STAMP=$(date -u +%Y%m%dT%H%MZ)`** + **`M1_LEAVES="8"`** **`bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`** | JSON 拷入 **`results/metrics_result/`**；**`EXPERIMENT_REGISTRY`** **X-ssgs-vs-kv-tree-nav-m1** 附 **新 STAMP** 一句 |
| **B2** | **M1 · n64 + L3 下游 CE** — **环节 ②+③**；**非**新对比法。**本仓已有** **`…_n64_cuda_3arm_20260410T1247Z.json`** 时以 **复现/刷新** 为主（**全链条 smoke** 仍建议先做 **§6.0** 步 1） | **`M1_LEAVES="64" M1_WITH_L3_DOWNSTREAM_CE=1`** **`bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`** | 跑后 **`aggregate_ssgs_vs_kv_wikitext_json.py`**；核对 **`abs_ce_delta`**；新 **STAMP** → JSON 拷回仓 + **登记册** |
| **B3** | **M1 · chunk_len 消融**（与 **A2-S2** **c12** 叙事对齐，**分列**） | 脚本默认 **c8**；改 **`--chunk-len 12`** 须 **直接调用** **`python scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py`**（参数同 **`run_m1_*.sh`**，改 **`--chunk-len`** / **`--out-json`** 带新 **STAMP**） | **新 basename**；正文写 **「M1 玩具 TF-KV 臂；chunk_len=12；与 path-batch 表分列」** |
| **B4** | **SSGS 辅线 · 新格点**（非 M1） | **`docs/environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md` §10** **`run_ssgs_mamba_wikitext_cuda.sh`**；叶数 / **`STAMP`** 自定 | **`aggregate_ssgs_mamba_wikitext_json.py`** → **`ssgs_mamba_wikitext_grid.csv`** |

### 6.3 Wave **C** — 主曲线延伸（**另排期**；**非** 投稿默认阻塞）

- **path-batch** 与 **SSGS** **同树 bundle** 扩展（**`RUN_WIKITEXT_SMOKE=1`** 已有入口，见 **`run_ssgs_mamba_wikitext_cuda.sh`** 注释）。  
- **更深树 / RAPTOR 式建树**：新 **`kind`** 或新脚本 — 见 **`PROJECT_MASTER_PLAN.md` §2**、**`NEXT_RESEARCH_PLAN.md` §0.2**「阶段 2 延伸」。  
- **dim256 M1**：显存与实现成本显著上升 — **仅**在假设明确时开 **独立 STAMP** 网格。

### 6.4 Wave **D** — **P★**（默认 **不做**）

- **训练型 L3**、**可学习导航抬 reach_rate**、**B-S3** — 见 **`RESEARCH_STATUS` §3.5**、**`PROJECT_MASTER_PLAN` §1.0** 副线；**须新 `kind`**，**禁止**与当前 M1 表无脚注合并。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-10 | 初版：工具齐全性盘点、缺口、M1 目标与检查表 |
| 2026-04-07 | 增补 **`run_m1_ssgs_vs_kv_wikitext_cuda.sh`**（叶数扫）；检查表：**叶数扩展** + **L3 探针** |
| 2026-04-10 | **叶扫归档**：**`20260410T1012Z`** **n8/n16/n32** 三臂 JSON → **`EXPERIMENT_REGISTRY` M1** 行 |
| 2026-04-10 | **§2.1** 实测表；**L3 隐状态**：**`tf_kv_l3_probe`**、**`--l3-tf-kv-hidden`** |
| 2026-04-10 | **网格 CSV**：**`aggregate_ssgs_vs_kv_wikitext_json.py`**、**`ssgs_vs_kv_wikitext_nav_grid.csv`**；检查表勾选 |
| 2026-04-10 | **L3 下游 CE**：**`--l3-tf-kv-downstream-ce`**、**`tf_kv_l3_downstream_probe.py`**；脚注 **X-20260423/24** 叙事参考 |
| 2026-04-10 | **归档**：**`STAMP=20260410T1113Z`** **n8** 三臂 + **`l3_tf_kv_downstream_ce`**（**`…_n8_cuda_3arm_20260410T1113Z.json`**） |
| 2026-04-11 | **归档**：**`STAMP=20260410T1133Z`** **n16/n32** **`l3_tf_kv_downstream_ce`**（**`abs_ce_delta`=0**）；§10.1 推荐命令已跑通 |
| 2026-04-11 | **索引**：**`docs/README`**、**`CURRENT_SPRINT`**、**`ROADMAP`**、**`NEXT_EXPERIMENTS`** 篇首、**`SERVER_SWEEP_RUNBOOK`** — **七轴** 与 **P0** **M1** 入稿指针 |
| 2026-04-11 | **§6 M2 跑道**：Wave A–D；检查表 **§3.5 / pytest** 与 **AutoDL** 对齐；**B2** **n64+L3 CE**、**B3** **chunk_len** 直调 **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** |
| 2026-04-11 | **§6.0**：**B2** vs **全链条**（①②③④）；AutoDL **先 n64 三臂 smoke 再 B2**；**path-batch smoke** 指针 |
| 2026-04-11 | **§6.0 / B2**：与 **`1247Z`** **n64 L3 CE** 已入仓一致 — **B2** = 复现或补档 |
| 2026-04-11 | **§2.1**：**n64** 三臂复跑 **`20260410T1617Z`**；登记与 **`NEXT_EXPERIMENTS` §12** 终端粘贴提示 |
| 2026-04-11 | **§2.1**：**1616Z**（无 L3 CE）/ **1617Z**（**+** L3 CE **`abs_ce_delta`=0**）分列；**grid** 仓内 **`aggregate_*`** |
