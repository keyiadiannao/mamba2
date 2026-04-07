# 实验：玩具树 + Reader 微基准（阶段 1 入口）

- **registry id**: `A-20260407-toy-tree-reader-bench`
- **方向**: A（树状 RAG 框架 / 验证实验）
- **目的**: 在**同一棵树、同一路径批量**上对比 **Transformer 路径编码器** 与 **GRU（Mamba 占位）** 的耗时与 `peak_alloc`；后续将 GRU 换成 Mamba-2 reader。

## 命令

仓库根目录、`conda activate mamba2`：

```powershell
python scripts\benchmark_tree_walk.py --depth 6 --fanout 2 --reps 10
```

可选写出 JSON（建议设 `MAMBA2_RESULTS_ROOT`）：

```powershell
python scripts\benchmark_tree_walk.py --depth 6 --fanout 2 --out-json results\metrics\A-20260407-bench.json
```

## 参数说明

- `--depth` / `--fanout`：完全 k 叉树，叶数 = `fanout**depth`；`depth=6,fanout=2` → 64 条根到叶路径。
- `--chunk-len`：每个节点合成「块」长度（沿路径拼接成序列）。
- `--dim`：需被 `--nhead` 整除（默认 128 / 8）。

## 结论（本地一次样例）

在 `depth=4, fanout=2`、5060 cu128 上，同 batch 下 GRU 步均时间常低于 Transformer（短序列下 attention 开销仍可见）；**峰值显存**以脚本打印 `peak_alloc_mib` 为准，随叶数与 `dim` 扫描可画曲线（阶段 1 主表雏形）。
