# 当前迭代（滚动）

> 每 1–2 周更新一次「周期」与勾选；完成后把结论一行写入 `docs/experiments/EXPERIMENT_REGISTRY.md`。

## 周期

**开始**：2026-04-07  
**当前滚动至**：2026-04-21 起（主文级 3090 数据已齐，迭代重点转向叙事收束与 Wikitext/§7 边界）

---

## 本迭代目标

把阶段 1 从**纯合成树**推进到**可读的文本形树 + 同一 reader 基准槽位**；**AutoDL** 上 **fused 主环境**与 **`mamba2_naive` 同机对照**已跑通（见 `EXPERIMENT_REGISTRY` **A-20260408-paper-main-3090-***）。

---

## 任务清单

- [x] 总体规划文档 `docs/overview/PROJECT_MASTER_PLAN.md`
- [x] 扫参 CSV 增强：`gpu_name`、`torch_version`；合并多机 CSV 脚本 `scripts/benchmarks/merge_sweep_csv.py`
- [x] **文本形浅树**：样例叶文本 + 自底向上建树 + `scripts/benchmarks/benchmark_text_tree.py` + `run_reader_benchmark_on_paths`（确定性嵌入，非神经 encoder）
- [x] **数据约定**：`data/raw/sample/` 8 段合成 `.txt` + `docs/experiments/DATASETS.md`；`scripts/data/prepare_leaves_from_corpus.py` 生成叶文件
- [x] **AutoDL 文档**：`docs/environment/AUTODL_SETUP.md` + `SYNC` 索引；**已在 3090 实例跑通** smoke + `sweep_autodl.csv`（见 `EXPERIMENT_REGISTRY`）
- [x] **本地最小 Mamba**：`transformers.MambaModel` 小配置 smoke（无需 `mamba-ssm`），见 `scripts/smoke/smoke_mamba_minimal.py`
- [x] （可选）**mamba-ssm**：AutoDL 已装；**同机 naive 对照**见 `run_server_paper_main_sweep_naive.sh` + `SERVER_SWEEP_RUNBOOK` §2c（`mamba2_naive` 克隆环境卸融合栈）
- [x] **树路径三 reader**：`Mamba2PathReader` 接入 `benchmark_core` / `sweep` / `benchmark_text_tree`（默认开启，`--no-mamba2` 可关）
- [x] **公开语料浅树**：Wikitext-2（HF `datasets`）→ `hf_corpus.wikitext2_leaf_chunks` → `scripts/benchmarks/benchmark_wikitext_tree.py`（与合成叶同一 reader 槽位）
- [x] **3090 主文扫参（同机）**：`run_server_paper_main_sweep.sh`（fused）+ `run_server_paper_main_sweep_naive.sh`（`mamba2_naive`）；CSV 归档示例见本机 `results/metrics_result/`；图 `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`
- [x] **3090 Wikitext 浅树（fused + 镜像）**：`benchmark_wikitext_tree.py`；`results/metrics_result/benchmark_wikitext_3090_fused_20260408T0846Z.json`；登记 **A-20260408-wikitext-3090-fused**
- [x] **主文图注模板**：`docs/experiments/FIGURE_CAPTIONS_STAGE1.md`
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
- [x] **树上导航任务指标（启发式）**：`tree_lm_nav_eval.py` + `demo_tree_lm_nav_greedy.py`（子文档 CE argmin、**reach_rate** / **child_choice_accuracy**）；登记 **X-20260423-tree-lm-nav-greedy**；下一步：**可学习子选择头** 或 **与 SSGS 轨迹对照**

---

## 阻塞项

- **云端算力**：**按需开机**即可；主环境 **`conda activate mamba2`** 下 fused 已验证。**检索头训练**等仍受机时与预算约束，非每日阻塞。
- **仅本机 5060 时**：无法复现 3090 fused 数字；以登记册与图为准，不混填表格。

---

## 优先事项（本机 + 云端）

| 优先级 | 方向 | 可执行项 |
|--------|------|----------|
| P0 | **阶段 1 叙事收束** | **`FIGURE_CAPTIONS_STAGE1.md`**（篇首 **P0 叙事边界（中文摘要）** + 中英图注/表注）；`RESEARCH_NOTES.md` **§7.0** 段首同主题摘要 |
| P0 | **真实语料线（云端）** | （已完成）3090 + `HF_ENDPOINT` 镜像；JSON 与登记见 **A-20260408-wikitext-3090-fused** |
| P1 | **测量协议成文** | S1–S4；**真 LM**：**X-20260422** + **X-20260423**（`demo_tree_lm_nav_greedy.py` **reach_rate** 等）；下一跳：**可学习导航** / **SSGS×LM** 组合 |
| P1 | **SSGS** | 张量玩具 JSON：**X-20260421-ssgs-tensor-overhead-fixed**；**Mamba `DynamicCache` DFS**：`dfs_ssgs_mamba` + `demo_ssgs_mamba_dfs.py --out-json`（**X-20260421-ssgs-mamba-dfs-demo**，见 `RESEARCH_NOTES` §6 / §7.4） |
| P2 | **检索头** | 读论文与接口草图；训练探针等 **GPU 空闲窗口** |
| 延后 | **大叶数研究扫参** | （已完成）**A-20260408-research-large-leaves-3090**；扩展网格可另开新 TAG |

**讨论结论（写入此表的目的）**：主故事仍是 **`docs/overview/PROJECT_MASTER_PLAN.md` §1.1 树内 Mamba vs Transformer**；**状态快照 / SSGS** 作为协议层贡献，需 **公平基线下的曲线** 支撑，避免仅停留在类比。

---

## 上迭代归档（简述）

- 环境 `mamba2`、cu128、玩具树基准、本地 preset 扫参 8 点 CSV。
- **2026-04-08**：3090 `paper_main_v1` / `paper_main_naive_v1` 成对数据与同机三张 `mamba_3090_naive_vs_fused_*.png`（见 `EXPERIMENT_REGISTRY`）。
