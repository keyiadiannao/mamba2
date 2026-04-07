# 阶段 1 验证：规划与扫参说明

> 与 `PROJECT_OVERVIEW.md` 阶段 1 对齐；执行记录进 `EXPERIMENT_REGISTRY.md`，原始表格进 `results/metrics/`。

---

## 1. 目标（本阶段要回答什么）

1. **系统**：在「同一棵树、同一批根→叶路径、同一训练式步进（forward+backward）」下，**Transformer 路径编码** 与 **递归式路径编码（当前为 GRU，后续换 Mamba-2）** 的 **耗时** 与 **峰值显存** 如何随 **树规模**、**每节点序列长度** 变化。
2. **叙事**：若递归式在 **长路径 / 多叶 batch** 下相对更稳或更省（或反之），为论文「树导航 + 状态模型」提供 **可画图的数据**；若差异不大，则提前收紧假设或改评测设定（例如更长序列、更大 `dim`）。
3. **工程**：固定可复现脚本与 CSV 格式，便于 **5060 本地扫参** 与 **AutoDL 48G 补点**（更大 `depth` / `fanout`）。

---

## 2. 当前 harness 范围（已实现）

- **树**：完全 `fanout` 叉、深度 `depth`，叶数 \(L = \texttt{fanout}^{\texttt{depth}}\)。
- **节点表示**：每节点 `chunk_len × dim` 合成张量；路径序列长度 \(T = (\texttt{depth}+1)\times\texttt{chunk_len}\)。
- **Reader**：
  - **Transformer**：`TransformerEncoder` + mean pool + 线性头。
  - **GRU**：双层 GRU + 线性头（**Mamba 占位**）。
- **指标**：`elapsed_s`、`per_step_s`（总时间 / `reps`）、`peak_alloc_mib`（CUDA `max_memory_allocated`，CPU 为 0）。

**未纳入（后续迭代）**：真实 RAPTOR 文本、Mamba-2、`mamba_ssm`、检索头、任务级 EM/F1。

---

## 3. 扫参维度与推荐网格

| 维度 | 含义 | 本地 5060（建议先跑） | AutoDL（可选补全） |
|------|------|------------------------|---------------------|
| `depth` | 根到叶边数；叶数指数增长 | `3–7`，步长 1；注意 `2**7=128` 叶 | 可试 `8–10`（先设 `--max-leaves`） |
| `fanout` | 分叉数 | 固定 `2` | `2` 或 `3`（叶数爆炸快） |
| `chunk_len` | 每节点 token 列长度 | `4, 8, 16` | 与本地一致便于对比 |
| `dim` / `nhead` | 宽度与头数 | 默认 `128/8` | 可增 `256/8` 看算力瓶颈 |
| `reps` / `warmup` | 计时重复与预热 | `warmup=2–3`，`reps=5–10` | 可略增 `reps` 减方差 |

**安全阀**：扫参脚本支持 `--max-leaves`（例如 `512`），跳过超过该叶数的组合，避免 OOM 或过久。

---

## 4. 产出物与命名

- **CSV**：`results/metrics/sweep_tree_reader_<YYYYMMDD>_<optional_tag>.csv`（或置于 `MAMBA2_RESULTS_ROOT/metrics/`）。
- **登记**：在 `EXPERIMENT_REGISTRY.md` 增加一行，注明 CSV 路径、`git` commit、`device` 名称。
- **可选**：同一次扫参附带 `jsonl` 每行一条完整 JSON（脚本 `--out-jsonl`）。

---

## 5. 成功判据（第一版，可改）

- 至少完成一张 **「叶数或路径 token 数 × 两 reader 的 per_step_s / peak_mib」** 表或图。
- 记录 **环境**：`environment/requirements-mamba2-lock.txt` + 扫参时的 `torch` 版本行（可写在 registry）。
- 若 **GRU（及日后 Mamba）相对 Transformer** 在 **更长 `T`** 或 **更大 batch（叶数）** 上出现 **系统性差异**，阶段 1 可收束为「值得做系统对比」；否则进入 **真实语料 + 更长上下文** 再判。

---

## 6. 执行顺序（建议）

1. 跑默认小网格（脚本 `--preset local`）→ 检查 CSV 列完整、数值无 NaN。
2. 固定 `chunk_len=8`，扫 `depth`；再固定 `depth=6`，扫 `chunk_len`。
3. AutoDL 上复制同脚本、增大 `depth` 或 `dim`，合并 CSV（注明机器列已有 `device` 字符串）。
4. 安装 **Mamba-2** 后增加第三列 reader，重复同一网格（文档与脚本届时扩展）。

---

## 7. 命令速查

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2

# 单次
python scripts\benchmark_tree_walk.py --depth 6 --fanout 2 --out-json results\metrics\single.json

# 扫参（预设本地小网格）
python scripts\sweep_tree_benchmark.py --preset local --out-csv results\metrics\sweep_tree_reader_local.csv
```

---

## 8. 相关文件

| 文件 | 作用 |
|------|------|
| `src/rag_tree/benchmark_core.py` | 可编程入口 `run_tree_reader_benchmark` |
| `scripts/benchmark_tree_walk.py` | 单次 CLI |
| `scripts/sweep_tree_benchmark.py` | 多组 CSV / jsonl 扫参 |
| `experiments/A-20260407-toy-tree-reader-bench/README.md` | 单次实验说明 |
