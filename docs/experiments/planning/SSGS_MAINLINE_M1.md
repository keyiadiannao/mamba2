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

**文件**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n{8,16,32}_cuda_3arm_20260410T1012Z.json`**。另：**首轮无 STAMP** 的 **`…_n8_cuda_3arm.json`** 可作对照。

**趋势（一句）**：**Mamba** `peak_alloc_mib` 三档均 **≈130**；**TF-KV** **≈27.7–28.1**；**`wall_s`** 随叶数上升主要来自 **DFS 迹长度**，**truncate** 臂略快于 **clone** 臂（同 harness）。

**L3 隐状态（n8 CUDA）**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm_l3.json`** — **`l3_tf_kv_hidden`**：**clone / truncate** 与金路径-only 的 **末 token hidden** **余弦 ≈ 1**、**`l2_diff`=0**（**`git_sha` 以 JSON 为准**）。

**L3 下游 CE（n8 CUDA，`STAMP=20260410T1113Z`）**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm_20260410T1113Z.json`** — **`l3_tf_kv_downstream_ce`**（**`--l3-tf-kv-downstream-ce`**）：**clone** **`ce_nav`=`ce_ref`≈2.121**、**truncate** **`ce_nav`=`ce_ref`≈2.224**、两臂 **`abs_ce_delta`=0**、**`max_abs_logit_diff`=0**（**`probe_seed`=12345**）。同次跑 **Mamba** **`wall_s`≈0.507**、**clone**≈0.074、**truncate**≈0.031（与 **1012Z** 同阶，GPU 抖动正常）。

**待补（CUDA）**：**n16 / n32** 的 **`l3_tf_kv_downstream_ce`**（判据同 n8：**`abs_ce_delta`** 应极小）。云端：**`M1_LEAVES="16 32" M1_WITH_L3_DOWNSTREAM_CE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**；拷回 JSON → 本机 **`aggregate_ssgs_vs_kv_wikitext_json.py`** → 更新 **登记册** / 本节一句。步骤全文：**`NEXT_EXPERIMENTS_COMMANDS` §10.1**「推荐下一步」。

**网格 CSV**：**`results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`** — **`python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py -g 'results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json' --out-csv results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`**。云端 **`run_m1_ssgs_vs_kv_wikitext_cuda.sh`** 扫叶后默认聚合（**`SKIP_M1_AGGREGATE=1`** 跳过；**`AGGREGATE_APPEND=1`** 追加）。

---

## 3. Phase M1 目标（4 周内可检查）

1. **选定基线臂**：优先 **TF-KV 玩具 trunk 上的「错枝 + 截断 KV + 兄弟子树」**（与现有 **`--branch-truncate-demo`** 语义衔接），或 **「回到分叉点整段重算」** 二选一写死。  
2. **选定任务**：与现有 **SSGS** 一致的最右叶 / 或 **goal 叶**（与 **`demo_ssgs_mamba_wikitext`** 的 **`target_leaf_index`** 对齐）。  
3. **统一报告**：**总前向步数或 token 数**、**wall-clock**、**峰值显存**、**snapshots/rollbacks**（SSGS 臂）、**kv 字节或截断次数**（KV 臂）、**任务成功**（`ok`）。  
4. **登记**：**`EXPERIMENT_REGISTRY.md` 新行**；JSON 入 **`results/metrics_result/`**；**`FIGURE_CAPTIONS_STAGE1.md`** 若新增第六张「对照图」须 **显式第六轴或子表脚注**。

---

## 4. 开工检查表（复制到 issue / sprint）

- [ ] 读完 **`RESEARCH_STATUS_AND_DIRECTION.md` §3.5**（证据层级与风险三段）。  
- [ ] **`pytest tests/test_ssgs*.py`** + **`test_aggregate_ssgs_mamba_wikitext_json.py`** 绿。  
- [x] 在 **`EXPERIMENT_REGISTRY`** 登记 **M1**：**`X-ssgs-vs-kv-tree-nav-m1`** + **`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm.json`**（三臂，**`git_sha=6fa7873`**）；历史两臂文件 **`…_n8_cuda.json`** 仍保留作对照。  
- [x] **叶数扩展**：**`STAMP=20260410T1012Z`** — **`…_n{8,16,32}_cuda_3arm_20260410T1012Z.json`** 已入仓并写入 **`EXPERIMENT_REGISTRY`** **X-ssgs-vs-kv-tree-nav-m1**；**n64** 等仍可选。  
- [x] **L3（隐状态）**：**`--l3-tf-kv-hidden`** + **`src/rag_tree/tf_kv_l3_probe.py`**；单测 **`tests/test_tf_kv_l3_probe.py`**。  
- [x] **网格 CSV**：**`aggregate_ssgs_vs_kv_wikitext_json.py`** → **`ssgs_vs_kv_wikitext_nav_grid.csv`**；单测 **`tests/test_aggregate_ssgs_vs_kv_wikitext_json.py`**。  
- [x] **L3（下游 CE，固定叶头）**：**`--l3-tf-kv-downstream-ce`**；单测 **`tests/test_tf_kv_l3_downstream_probe.py`**。树 LM **可学习 vs CE**（**X-20260423/24**）为**另一 harness**，不作数值可比。  
- [ ] **L3 下游 CE · n16/n32（CUDA）**：**`M1_LEAVES="16 32" M1_WITH_L3_DOWNSTREAM_CE=1`** → 归档 JSON + 聚合 + **登记册** / §2.1（见 **§10.1 推荐下一步**）。  
- [ ] **L3（训练型子头 / 与树 LM 对齐）**：若要做，须另 **`kind`** 与登记；**禁止**与 path-batch 主表无脚注合并。

---

## 5. 相关文档

- **`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`** — 五轴、§3.5、L3 最小验证。  
- **`docs/overview/execution/NEXT_RESEARCH_PLAN.md`** — **M1** 与 **P0 成文 / P1 B-S2+ CUDA** 并行关系。  
- **`docs/research/RESEARCH_NOTES.md`** §6–§7 — SSGS 与 §7 边界。  
- **`scripts/README.md`** — 脚本索引（含 SSGS / §7）。  
- **M1 云端扫叶数**：**`scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**（默认 **`M1_LEAVES="8 16 32"`**、**三臂**；**`M1_NO_TRUNCATE=1`** 仅两臂）。

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
| 2026-04-11 | **待跑**：**n16/n32** **L3 下游 CE**；**`NEXT_EXPERIMENTS_COMMANDS` §10.1** 推荐命令块 |
