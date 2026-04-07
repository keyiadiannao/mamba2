# 实验：文本形浅树 + Reader 基准（A）

- **registry id**: `A-20260408-text-shaped-tree`
- **说明**: 叶节点为真实短句（UTF-8），自底向上合并父节点文本；节点向量由 **SHA256 种子 + RNG** 生成（**非**神经编码器），用于在「文本语义占位」下跑通与合成树相同的 **Transformer vs GRU** 计时。

## 命令

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2
python scripts\benchmark_text_tree.py
python scripts\benchmark_text_tree.py --leaf-file experiments\A-20260408-text-shaped-tree\leaves_sample.txt
```

叶行数须满足 `len = fanout ** depth`（默认 `fanout=2` → 1, 2, 4, 8, 16, …）。

## 下一步

- 将 `text_embedding` 换为小模型（如 `sentence-transformers` 或本仓库 encoder）时，保持建树与 `batched_paths` 不变，仅替换 `src/rag_tree/from_text.py` 中嵌入函数。
