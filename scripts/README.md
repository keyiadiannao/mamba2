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
| **Linux** | `benchmarks/run_server_sweep_aligned.sh` | 服务器 fused 对齐扫参（与本地 CSV 键对齐） |
| **Linux** | `benchmarks/run_server_paper_main_sweep.sh` | **同机主文扫参 fused**（统一 `WARMUP`/`REPS` + manifest） |
| **Linux** | `benchmarks/run_server_paper_main_sweep_naive.sh` | **同机主文扫参 HF naive**（需无 `mamba-ssm`/`causal-conv1d` 的环境；网格与上者相同） |

示例：

```powershell
python scripts\smoke\smoke_local.py
python scripts\benchmarks\benchmark_tree_walk.py --depth 6 --fanout 2
```
