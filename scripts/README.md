# 脚本索引

仓库根目录执行；路径相对于 `mamba2/`。

| 子目录 | 脚本 | 说明 |
|--------|------|------|
| **[smoke/](smoke/)** | `smoke_local.py` | PyTorch / CUDA 冒烟 |
| | `smoke_mamba_minimal.py` | HF `Mamba2Model` / `MambaModel` 小配置前向 |
| **[benchmarks/](benchmarks/)** | `benchmark_tree_walk.py` | 玩具树路径 reader 微基准 |
| | `benchmark_text_tree.py` | 文本形叶 + 自底向上建树 |
| | `benchmark_wikitext_tree.py` | Wikitext-2 叶块 + 同 harness；**`--out-json`**、**`git_sha`** / **`torch_version`** |
| | `sweep_tree_benchmark.py` | 扫参 CSV / jsonl |
| | `merge_sweep_csv.py` | 多机 CSV 合并 |
| | `plot_tree_reader_sweep.py` | 扫参 CSV → 延迟/显存图（需 `matplotlib`） |
| | `plot_mamba_naive_vs_fused.py` | 两份 CSV 叠 Mamba2 `m2_peak_mib`（跨机仅示意） |
| | `aggregate_wikitext_5060_cuda_grid.py` | 5060 CUDA Wikitext **2×2** JSON → 扁平 CSV（`benchmark_wikitext_5060_cuda_grid_*.csv`） |
| | `aggregate_wikitext_tree_json_grid.py` | 任意 **`benchmark_wikitext_*_n*_c*.json`** → 一表 CSV（**A2-S2** 与 **`run_server_stage2_wikitext_grid.sh`** 配套） |
| | `benchmark_ssgs_tensor_overhead.py` | SSGS 张量快照/恢复微基准（无 LM）；`--out-json` 写登记用 JSON |
| **[data/](data/)** | `prepare_leaves_from_corpus.py` | 从目录生成叶文本文件 |
| **[sync/](sync/)** | `sync_example.ps1`, `sync_example.sh` | 双机同步命令模板 |
| **[research/](research/)** | `probe_mamba2_outputs.py` | §7.5 S1：探针 `Mamba2Model` 的 `forward` 输出字段（含可选 `use_cache`） |
| | `probe_retrieval_correlation.py` | **B-S2**：层 mean-pool **岭二分类**；`marker` / `digit` / **`topic`**（**`--topic-split heldout`** 默认）；`--out-json` |
| | `probe_path_reader_linear.py` | **B-S2+**：默认 **16 叶**、**heldout**；**ridge_untrained**；**`--train-steps`** + **`bce_reader_train`** 或 **`--train-head-only`**（**`bce_head_only_train`**）；`--n-leaves 8` |
| | `benchmark_mamba2_cache_snapshot_segments.py` | §7.5 S1：单路径上每多读一个节点对**累积** `inputs_embeds` 整段前向，边界处 **clone cache** nbytes/ms（段间不传 cache，兼容 fused CUDA） |
| | `benchmark_tf_r1_path_segments.py` | §7.5 S2 / §7.2 **TF-R1**：同路径设定下 `TransformerPathReader` **仅前向**、无 KV；每边界 `forward_mean_ms` + CUDA `peak_alloc_mib` |
| | `benchmark_tf_kv_path_segments.py` | §7.5 S3 / §7.2 **TF-KV**：Pre-LN 因果 trunk + MHA KV cache；`kv_cache_nbytes`、`increment_last_chunk_mean_ms`；可选 `--branch-truncate-demo` |
| | `benchmark_mamba2_cache_restore_segments.py` | §7.5 S4 / §7.3 **SSM restore**：S1 同款 cache 上 `zero_`→`copy_` 快照；`restore_wall_ms`；`--snapshot-device cpu` 含 H2D |
| | `demo_ssgs_mamba_dfs.py` | **SSGS × Mamba**：`dfs_ssgs_mamba` 玩具树 DFS（最右叶）；`MambaNavState` + token 前向 |
| | `demo_ssgs_mamba_wikitext.py` | **SSGS × Mamba × Wikitext**：与 **`benchmark_wikitext_tree`** **同建树**；**`--out-json`**；CUDA 上 `torch_forward` patch；登记 **X-20260407** |
| | `aggregate_ssgs_mamba_wikitext_json.py` | 多份 **`ssgs_mamba_wikitext_tree` JSON** → **`ssgs_mamba_wikitext_grid.csv`**；支持 **`--append`** |
| | `demo_tree_lm_minimal.py` | **真因果 LM 最小闭环**：文本形树 → 路径文档 → HF ``AutoModelForCausalLM`` CE + 续写；可选一步 ``AdamW`` |
| | `demo_tree_lm_nav_greedy.py` | **树上导航任务（启发式）**：每节点对各子「walk+子」文档算 CE，argmin 贪心；**--eval-all-leaves** → **reach_rate** / **mean_child_choice_accuracy** |
| | `demo_tree_lm_nav_learned.py` | **目标叶条件可学习子指针**：冻结 LM，**h_last + goal 叶嵌入** → 子 logits；监督 CE；**--eval-all-leaves**；登记 **X-20260424**（与 **X-20260423** 边界：**goal** vs 盲 argmin） |
| | `demo_ssgs_lm_nav_compare.py` | **SSGS×LM 玩具并列**：同 **X-20260424** 文本 8 叶树；每 goal：**dfs_ssgs_mamba**（必达）vs **子头贪心**；**--out-json**；登记 **X-20260425** |
| | `task_wikitext_path_pair.py` | **A2-S3**：Wikitext 浅树 **叶对同 cohort** 二分类（ridge / concat）；**`--pair-split leaf_heldout`**；归档 **`results/metrics/task_wikitext_*.json`** |
| **Linux** | `research/run_path_protocol_cuda.sh` | §7 玩具 S1–S4 串行 CUDA 复跑 → `metrics/`（见 `RESEARCH_NOTES` §7.3.1） |
| **Linux** | `benchmarks/run_server_sweep_aligned.sh` | 服务器 fused 对齐扫参（与本地 CSV 键对齐） |
| **Linux** | `benchmarks/run_server_paper_main_sweep.sh` | **同机主文扫参 fused**（统一 `WARMUP`/`REPS` + manifest） |
| **Linux** | `benchmarks/run_server_paper_main_sweep_naive.sh` | **同机主文扫参 HF naive**（需无 `mamba-ssm`/`causal-conv1d` 的环境；网格与上者相同） |
| **Linux** | `benchmarks/run_server_stage2_wikitext_grid.sh` | **A2-S2**：Wikitext 浅树 **fused** 四格（或 **`GRID=minimal`**）；见 **`SERVER_SWEEP_RUNBOOK` §2d** |
| **Linux** | `benchmarks/run_server_wikitext_dim256_grid.sh` | Wikitext **dim=256** 同拓扑四格（**`TAG` 默认 `stage2_dim256`**）；见 **`NEXT_EXPERIMENTS_COMMANDS.md`** |
| **Linux** | **[server/bootstrap_autodl.sh](server/bootstrap_autodl.sh)** | AutoDL：**conda mamba2** + **`HF_ENDPOINT`** + **`MAMBA2_RESULTS_ROOT`** + Wikitext/datasets 与 **CUDA** 冒烟 |
| **Linux** | **[server/run_ssgs_mamba_wikitext_cuda.sh](server/run_ssgs_mamba_wikitext_cuda.sh)** | 主线：**SSGS × Mamba × Wikitext 同建树**（`demo_ssgs_mamba_wikitext.py` **CUDA**）+ 可选 path-batch smoke + **`aggregate_ssgs_mamba_wikitext_json.py`** |
| **Linux** | **[server/run_m1_ssgs_vs_kv_wikitext_cuda.sh](server/run_m1_ssgs_vs_kv_wikitext_cuda.sh)** | **Phase M1**：**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** **CUDA**；默认 **n8/n16/n32** **三臂**（可 **`M1_LEAVES`** / **`M1_NO_TRUNCATE=1`**） |

示例：

```powershell
python scripts\smoke\smoke_local.py
python scripts\benchmarks\benchmark_tree_walk.py --depth 6 --fanout 2
```
