# A-20260410-wikitext-shallow-tree

**目的**：在**真实公开文本**（Wikitext-2 raw train）上建浅层 bottom-up 树，复用 `run_reader_benchmark_on_paths`（TF / GRU / Mamba2）。

**命令**：

```bash
pip install datasets   # 若未装
python scripts/benchmarks/benchmark_wikitext_tree.py --num-leaves 8 --fanout 2 --chars-per-leaf 600
```

**登记**：见 `docs/experiments/planning/EXPERIMENT_REGISTRY.md` 对应行。
