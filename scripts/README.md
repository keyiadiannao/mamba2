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
| **[data/](data/)** | `prepare_leaves_from_corpus.py` | 从目录生成叶文本文件 |
| **[sync/](sync/)** | `sync_example.ps1`, `sync_example.sh` | 双机同步命令模板 |

示例：

```powershell
python scripts\smoke\smoke_local.py
python scripts\benchmarks\benchmark_tree_walk.py --depth 6 --fanout 2
```
