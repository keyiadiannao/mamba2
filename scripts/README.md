# 脚本索引

仓库根目录执行；路径相对于 `mamba2/`。

| 子目录 | 脚本 | 说明 |
|--------|------|------|
| **[smoke/](smoke/)** | `smoke_local.py` | PyTorch / CUDA 冒烟 |
| | `smoke_mamba_minimal.py` | HF `Mamba2Model` / `MambaModel` 小配置前向 |
| **[benchmarks/](benchmarks/)** | `benchmark_tree_walk.py` | 玩具树路径 reader 微基准 |
| | `benchmark_text_tree.py` | 文本形叶 + 自底向上建树 |
| | `benchmark_wikitext_tree.py` | Wikitext-2 叶块 + 同 harness |
| | `sweep_tree_benchmark.py` | 扫参 CSV / jsonl |
| | `merge_sweep_csv.py` | 多机 CSV 合并 |
| | `plot_tree_reader_sweep.py` | 扫参 CSV → 延迟/显存图（需 `matplotlib`） |
| | `plot_mamba_naive_vs_fused.py` | 两份 CSV 叠 Mamba2 `m2_peak_mib`（跨机仅示意） |
| | `benchmark_ssgs_tensor_overhead.py` | SSGS 张量快照/恢复微基准（无 LM）；`--out-json` 写登记用 JSON |
| **[data/](data/)** | `prepare_leaves_from_corpus.py` | 从目录生成叶文本文件 |
| **[sync/](sync/)** | `sync_example.ps1`, `sync_example.sh` | 双机同步命令模板 |
| **[research/](research/)** | `probe_mamba2_outputs.py` | §7.5 S1：探针 `Mamba2Model` 的 `forward` 输出字段（含可选 `use_cache`） |
| | `benchmark_mamba2_cache_snapshot_segments.py` | §7.5 S1：单路径上每多读一个节点对**累积** `inputs_embeds` 整段前向，边界处 **clone cache** nbytes/ms（段间不传 cache，兼容 fused CUDA） |
| | `benchmark_tf_r1_path_segments.py` | §7.5 S2 / §7.2 **TF-R1**：同路径设定下 `TransformerPathReader` **仅前向**、无 KV；每边界 `forward_mean_ms` + CUDA `peak_alloc_mib` |
| | `benchmark_tf_kv_path_segments.py` | §7.5 S3 / §7.2 **TF-KV**：Pre-LN 因果 trunk + MHA KV cache；`kv_cache_nbytes`、`increment_last_chunk_mean_ms`；可选 `--branch-truncate-demo` |
| **Linux** | `benchmarks/run_server_sweep_aligned.sh` | 服务器 fused 对齐扫参（与本地 CSV 键对齐） |
| **Linux** | `benchmarks/run_server_paper_main_sweep.sh` | **同机主文扫参 fused**（统一 `WARMUP`/`REPS` + manifest） |
| **Linux** | `benchmarks/run_server_paper_main_sweep_naive.sh` | **同机主文扫参 HF naive**（需无 `mamba-ssm`/`causal-conv1d` 的环境；网格与上者相同） |

示例：

```powershell
python scripts\smoke\smoke_local.py
python scripts\benchmarks\benchmark_tree_walk.py --depth 6 --fanout 2
```
