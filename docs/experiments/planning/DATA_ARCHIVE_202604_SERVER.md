# 数据归档索引 · 2026-04 服务器批次（3090 / AutoDL）

> **用途**：把 **`results/metrics_result/`** 下 **20260410** 前后入仓的 JSON 与 **汇总 CSV**、**`EXPERIMENT_REGISTRY`** 行 **对齐**，避免只记 STAMP 不记路径。  
> **代码提交**（本批 JSON 内 **`git_sha` 短哈希**）：**`6fa7873`**（与 manifest **全 SHA** **`6fa78733cf6b75d2f9a492c73c65503bedba5b2a`** 一致）。  
> **聚合**：在**仓库根**执行 **`aggregate_ssgs_vs_kv_wikitext_json.py`** / **`aggregate_ssgs_mamba_wikitext_json.py`**；**`json_path`** 列为 **`results/metrics_result/…`（POSIX 相对路径）**。

---

## 0. 核心 JSON 速查（与 **CSV** 并列的「单行登记」文件）

| Basename（均在 **`results/metrics_result/`**） | 登记 id（**`EXPERIMENT_REGISTRY`**） |
|-----------------------------------------------|--------------------------------------|
| **`benchmark_wikitext_headcheck_20260410T1231Z_n8_c8.json`** | **X-20260410-benchmark-wikitext-headcheck-3090** |
| **`probe_path_reader_linear_text16_heldout_train50_cuda_20260410T1302Z.json`** | **X-20260410-probe-path-reader-bs2plus-cuda-3090** |
| **`tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`** | **X-20260411-tf-kv-trajectory-l3-minimal** |
| **`ssgs_vs_kv_tree_nav_wikitext_*.json`**（多 **STAMP**） | **X-ssgs-vs-kv-tree-nav-m1** |
| **`ssgs_mamba_wikitext_*.json`**（多 **STAMP**） | **X-20260407-ssgs-mamba-wikitext-tree** |
| **`benchmark_wikitext_stage2_leavescale_20260410T1240Z_n*_c8.json`**（×4）+ grid/manifest | **X-20260410-stage2-leavescale-rerun-3090** |

**论文叙事**：以上与 **path-batch 主 CSV**、**§7 JSON**、**5060 动机 JSON** 仍须 **分列脚注**；完整核对表见 **`SUBMISSION_PACK.md` §A2**。

---

## 1. 汇总表（CSV）

| 文件 | 行数（量级） | 说明 |
|------|----------------|------|
| **`results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`** | **N**（**数据行**；**N** = **`aggregate_ssgs_vs_kv_wikitext_json.py` stdout**；随通配 JSON 数变；**2026-04-11** 本机重建 **18**；+1 表头） | **M1** 三臂 + 多 **STAMP**（含 **`chunk_len=12`** **M2** 一条）；**`json_path`** 宜为 **`results/metrics_result/…`**（仓根 **`aggregate_*`** 重写） |
| **`results/metrics_result/ssgs_mamba_wikitext_grid.csv`** | **16** | **SSGS × Mamba** 同 Wikitext 建树；含 **n128**；**2026-04-11** **`…T0204Z` / `…T0303Z` / `…T0334Z`** **n8** 复跑（迹 **7/11/8**） |
| **`results/metrics_result/benchmark_wikitext_stage2_leavescale_grid_20260410T1240Z.csv`** | **12** | **path-batch** **n∈{8,16,32,64}** **c8 dim128**（与 **20260409T1257Z** 档 **分列**） |

---

## 2. 按主题 · 原始 JSON（主要 STAMP）

### 2.1 Phase **M1**（**`kind=ssgs_vs_kv_tree_nav_wikitext`**）

| STAMP / 说明 | 叶数 | 备注 |
|----------------|------|------|
| **`20260410T1012Z`** | n8, n16, n32 | 三臂基准扫 |
| **`20260410T1113Z` / `T1133Z`** | n8；n16, n32 | **L3 下游 CE**（**`abs_ce_delta`=0**） |
| **`20260410T1235Z`** | **n64** | 三臂、**无** L3 块 |
| **`20260410T1244Z`** | n8 | **L3 隐状态**（**`l3_tf_kv_hidden`**） |
| **`20260410T1247Z`** | n8, n16, n32, **n64** | **L3 下游 CE** 全档 |
| **`20260410T1616Z`** | **n64** | 三臂、**无** **`l3_tf_kv_downstream_ce`** |
| **`20260410T1617Z`** | **n64** | 三臂 **+ L3 下游 CE**（**`abs_ce_delta`=0**；**`git_sha`/墙钟** 以 JSON 为准） |
| **`20260411T0202Z`** | **n8** | **`chunk_len=12`**（**M2 §Ⅲ-1**）；与 **c8** 默认档 **分列** |

**通配**：**`results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json`**（另含历史无 STAMP / 旧 STAMP 文件时，以 **`aggregate` 实际合并行** 为准）。

**登记**：**`EXPERIMENT_REGISTRY.md`** **`X-ssgs-vs-kv-tree-nav-m1`**。详表与脚注：**`SSGS_MAINLINE_M1.md`** §2.1。

### 2.2 **SSGS × Mamba × Wikitext**

| 文件（示例） | 说明 |
|--------------|------|
| **`ssgs_mamba_wikitext_n128_c8_dim128_cuda_20260410T1238Z.json`** | **n128**（**127 / 247 / 128**） |
| **`ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260410T1238Z.json`** | 同会话 **n8** 刷新 |
| **`ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0204Z.json`** | **n8** 与 **grid** 对齐 CUDA 复跑（迹 **7/11/8**） |
| **`ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0303Z.json`** | **n8** 再次复跑（**`git_sha=6fa7873`**；与 **0204Z** 同阶迹） |
| **`ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0334Z.json`** | **n8** 再次复跑（**`STAMP=0334Z`**；与上同阶迹） |

**登记**：**`X-20260407-ssgs-mamba-wikitext-tree`**。

### 2.3 **path-batch · Wikitext leavescale**

| 文件 | 说明 |
|------|------|
| **`benchmark_wikitext_stage2_leavescale_20260410T1240Z_n{8,16,32,64}_c8.json`** | 四 JSON |
| **`benchmark_wikitext_stage2_leavescale_grid_20260410T1240Z.csv`** | 汇总 |
| **`benchmark_wikitext_stage2_leavescale_manifest_20260410T1240Z.txt`** | manifest |

**登记**：**`X-20260410-stage2-leavescale-rerun-3090`**（与 **A-stage2-wikitext-leavescale-v1** **`1257Z`** **分列**）。

### 2.4 **HEAD / smoke**

| 文件 | 说明 |
|------|------|
| **`benchmark_wikitext_headcheck_20260410T1231Z_n8_c8.json`** | **path-batch** 单格 **n8 c8 dim128** |

**登记**：**`X-20260410-benchmark-wikitext-headcheck-3090`**。

### 2.5 **B-S2+ · CUDA（path reader 探针）**

| 文件 | 说明 |
|------|------|
| **`results/metrics_result/probe_path_reader_linear_text16_heldout_train50_cuda_20260410T1302Z.json`** | **n16 heldout**、**train 50 step**；与 **5060 CPU** **分列** |

**登记**：**`X-20260410-probe-path-reader-bs2plus-cuda-3090`**。

### 2.6 **阶段 C · L3 轨迹甲·乙**（**`tf_kv_trajectory_l3_minimal`**）

| 项 | 说明 |
|----|------|
| **脚本** | **`scripts/research/benchmark_tf_kv_trajectory_l3_minimal.py`** |
| **已归档 JSON** | **`results/metrics_result/tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`**（**3090 CUDA**，**`git_sha=6fa7873`**） |
| **摘要** | **clone** **`cosine_hidden_a_vs_b`≈0.99999988**；**truncate_kv** **≈0.99999994**；两臂 **`l2_diff_hidden_a_vs_b`=0** |
| **登记** | **X-20260411-tf-kv-trajectory-l3-minimal** |
| **单测** | **`tests/test_tf_kv_trajectory_l3.py`** |

---

## 3. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：M1 / SSGS / leavescale / headcheck / B-S2+ CUDA 索引；**聚合 CSV `json_path`** 约定 |
| 2026-04-11 | **§2.6**：**阶段 C** **L3 轨迹** 脚本与登记指针 |
| 2026-04-11 | **§2.6**：**CUDA** **`tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json`** 入仓 |
| 2026-04-11 | **§0**：核心 JSON 与登记 id **速查表** |
| 2026-04-11 | **§1 / §2.1**：**M1 grid** **N** 以 **`aggregate_*` stdout** 为准（**2026-04-11** 重建 **18**；随通配 JSON 数变）；**1616Z** / **1617Z** **n64** 分列；与 **`NARRATIVE` §9** 快照互链 |
