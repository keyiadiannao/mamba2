# 阶段 1 验证：规划与扫参说明

> 与 `docs/overview/PROJECT_OVERVIEW.md` 阶段 1 对齐；执行记录进 `docs/experiments/EXPERIMENT_REGISTRY.md`；主文 CSV / §7 复跑等归档见 **`results/metrics_result/`**。  
> **阶段 1 一页式存档 + §7 CUDA 复跑指令**：**`PHASE1_COMPLETE_SUMMARY.md`**。**可贴正文成稿**：**`PHASE1_MANUSCRIPT.md`**。

---

## 1. 目标（本阶段要回答什么）

1. **系统**：在「同一棵树、同一批根→叶路径、同一训练式步进（forward+backward）」下，**Transformer 路径编码** 与 **递归式路径编码（当前为 GRU，后续换 Mamba-2）** 的 **耗时** 与 **峰值显存** 如何随 **树规模**、**每节点序列长度** 变化。
2. **叙事**：若递归式在 **长路径 / 多叶 batch** 下相对更稳或更省（或反之），为论文「树导航 + 状态模型」提供 **可画图的数据**；若差异不大，则提前收紧假设或改评测设定（例如更长序列、更大 `dim`）。
3. **工程**：固定可复现脚本与 CSV 格式，便于 **5060 本地扫参** 与 **AutoDL 48G 补点**（更大 `depth` / `fanout`）。

**注**：课题级「主对照 / 平面 RAG 消融 / 是否必须绑 Transformer+平面」的取舍见 **`docs/overview/PROJECT_MASTER_PLAN.md` §1.1**；阶段 1 **只解决树内** reader 类型 harness，**不强制**在此阶段接平面 RAG 基线。

---

## 2. 当前 harness 范围（已实现）

- **树**：完全 `fanout` 叉、深度 `depth`，叶数 \(L = \texttt{fanout}^{\texttt{depth}}\)。
- **节点表示**：每节点 `chunk_len × dim` 合成张量；路径序列长度 \(T = (\texttt{depth}+1)\times\texttt{chunk_len}\)。
- **Reader**：
  - **Transformer**：`TransformerEncoder` + mean pool + 线性头。
  - **GRU**：双层 GRU + 线性头（递归占位）。
  - **Mamba-2**：`Mamba2PathReader`（`inputs_embeds` 喂 `Mamba2Model`）；无 `mamba-ssm` 时为 HF naive 实现。
- **指标**：`elapsed_s`、`per_step_s`（总时间 / `reps`）、`peak_alloc_mib`（CUDA `max_memory_allocated`，CPU 为 0）。

**未纳入（后续迭代）**：真实 RAPTOR 全流程、**全规模 LLM** 上的 KV 与 SSM 主循环接线、检索头、任务级 EM/F1。

**已另册（玩具协议，≠ 上文 path-batch 扫参 CSV）**：`RESEARCH_NOTES` §7 / §7.3.1 与登记 **X-20260421-*** — S1（Mamba `DynamicCache` clone）、S2（TF-R1 整段前向）、S3（TF-KV 增量 + 可选错枝截断）、S4（SSM `copy_` restore，same / CPU 快照）。原始 JSON 在 `results/metrics/*_20260421.json`；云端一键复跑：`bash scripts/research/run_path_protocol_cuda.sh`（`MAMBA2_RESULTS_ROOT` 可选）。

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
- **登记**：在 `docs/experiments/EXPERIMENT_REGISTRY.md` 增加一行，注明 CSV 路径、`git` commit、`device` 名称。
- **可选**：同一次扫参附带 `jsonl` 每行一条完整 JSON（脚本 `--out-jsonl`）。

---

## 5. 成功判据（第一版，可改）

- 至少完成一张 **「叶数或路径 token 数 × 两 reader 的 per_step_s / peak_mib」** 表或图。
- 记录 **环境**：`environment/requirements-mamba2-lock.txt` + 扫参时的 `torch` 版本行（可写在 registry）。
- 若 **GRU（及日后 Mamba）相对 Transformer** 在 **更长 `T`** 或 **更大 batch（叶数）** 上出现 **系统性差异**，阶段 1 可收束为「值得做系统对比」；否则进入 **真实语料 + 更长上下文** 再判。

---

## 6. 批判性收编与叙事限定（滚动，2026-04）

### 6.1 应接收的结论（有数据支撑）

- 在 **本仓库 path-reader harness**（`run_tree_reader_benchmark`：合成树、**批量根→叶路径**、小宽度 `Mamba2PathReader`）下，**无 `mamba-ssm` 的 HF 顺序实现（naive）** 的 **峰值显存与步耗** 可随 **叶数 batch** 急剧劣于同设定下的 **TransformerEncoder / GRU** 路径 reader（见 `sweep_tree_reader_20260410_local5060.csv`）。
- **Linux + fused 栈**（`mamba_ssm` + `causal_conv1d`）上，**同一脚本网格**可将 **Mamba2 峰值** 压到 **MiB 量级**（见 `sweep_autodl_fused.csv`）。这说明 **实现路径** 对可观测效率的影响极大，不能从「线性复杂度」口号直接推出「在树上一定省显存」。

### 6.2 不应过度外推的表述（写论文前须改口）

- 本阶段 **Transformer** 是 **小型 path encoder**，**不是** 长上下文 **LLM 全序列 KV cache** 对照；「比 KV 回溯便宜」仍属 **SSGS 协议层假说**（见 `docs/research/RESEARCH_NOTES.md` §7），须单独实验。
- **5060 与 3090、cu128 与 cu126、reps 不同** 时，两张 CSV 的对比是 **动机图 / 趋势图**；主文 **Figure 1** 应尽量 **同机、同 commit、同 `warmup/reps`** 复扫 naive vs fused。

### 6.3 阶段 1 结论文本（约 200 字，可贴进报告）

在树路径批量编码设定下（`fanout=2`、`dim=128`、多组 `depth×chunk_len`），HuggingFace **无融合核** 的 Mamba-2 path reader 的 CUDA **峰值显存** 可达 **GiB 级**（例如 64 条并行路径时约 **8.9GiB**），而同设定下 **Transformer / GRU** 路径 reader 多在 **百 MiB** 量级。相对地，在 **AutoDL** 上启用 **`mamba_ssm` 融合实现** 后，**同一网格**上 Mamba2 峰值可降至 **约 51–217MiB**（随配置变化），降幅可达 **两个数量级**。因此：**在树形 RAG 的 path-batch workload 中，可观测效率高度依赖实现是否融合，而不能仅由 SSM 架构名义复杂度替代。**

### 6.4 无服务器时本地仍可做的事（安全网格）

| 动作 | 说明 |
|------|------|
| **加密网格（小心显存）** | 固定 `dim=128`、`fanout=2`，用 `--max-leaves 64` 或 `32` 扫 `chunk_len`；**naive Mamba** 大叶数前先 `--no-mamba2` 探路 |
| **naive vs fused 示意图** | `scripts/benchmarks/plot_mamba_naive_vs_fused.py` 叠两张已有 CSV（图题写清跨机限定） |
| **叙事定稿** | 更新 `PROJECT_MASTER_PLAN` §1 贡献表述为「实现敏感 + workload 敏感」而非「Mamba 必然更省」 |

### 6.5 主文素材三角互证（登记 ↔ CSV ↔ 图，2026-04-09 审计）

**规则**：正文/幻灯片引用主文 **naive vs fused** 数字时，须同时指向 **`EXPERIMENT_REGISTRY`** 行与 **仓内路径**；**5060 本地 CSV** 与 **3090 主文 CSV** 不得混填为「同一点」。

| 登记 id | 含义 | 本仓库路径（存在性已核对） |
|---------|------|---------------------------|
| **A-20260408-paper-main-3090-fused** | 3090 fused 主文扫参 | `results/metrics_result/paper_main_dim128_localgrid_paper_main_v1.csv`、`paper_main_dim256_paper_main_v1.csv`、`paper_main_dim384_paper_main_v1.csv`；`paper_main_manifest_paper_main_v1.txt` |
| **A-20260408-paper-main-3090-naive** | 同网格 `mamba2_naive` | 同上 basename 后缀 **`_paper_main_naive_v1.csv`**；`paper_main_manifest_paper_main_naive_v1.txt` |
| **A-20260408-paper-main-3090-pair** | 同机成对对照（作图） | `results/metrics/figures/mamba_3090_naive_vs_fused_dim128_paper_main_v1.png`、`dim256`、`dim384` |
| **§7 复跑（STAMP）** | S1–S4 串行 JSON（与 `*_20260421.json` 同协议） | **`results/metrics_result/`** 内 `*_20260408T1617Z.json` 等（与 **`MAMBA2_RESULTS_ROOT`** 备份一致）；历史归档仍在 `results/metrics/*_20260421.json` |

**结论文本**：仍用 **§6.3**；写「同机 3090」时引 **pair** 行与上表 **PNG**。**§7 玩具协议**（S1–S4 ms）引 **X-20260421-*** 与 `run_path_protocol_cuda.sh`，**勿**与上表混为同一「一步」。**成稿叙述**见 **`PHASE1_MANUSCRIPT.md`**。

**§7 复跑（可选）**：有 CUDA 环境时执行串行脚本；**完整命令与环境说明**见 **`PHASE1_COMPLETE_SUMMARY.md` 附录 A**。无 GPU 时以 `results/metrics/*_20260421.json` 或 **`results/metrics_result/*_20260408T1617Z.json`** 为准。

---

## 7. 执行顺序（建议）

1. 跑默认小网格（脚本 `--preset local`）→ 检查 CSV 列完整、数值无 NaN。
2. 固定 `chunk_len=8`，扫 `depth`；再固定 `depth=6`，扫 `chunk_len`。
3. AutoDL 上复制同脚本、增大 `depth` 或 `dim`，合并 CSV（注明机器列已有 `device` 字符串）。
4. 安装 **Mamba-2** 后增加第三列 reader，重复同一网格（文档与脚本届时扩展）。

---

## 8. 与总体规划的关系

阶段 1 对应 `docs/overview/PROJECT_MASTER_PLAN.md` 中的 **阶段 1（系统验证）**；当前两周任务见 `docs/overview/CURRENT_SPRINT.md`。

---

## 9. 命令速查

```powershell
conda activate mamba2
cd d:\cursor_try\mamba2

# 单次
python scripts\benchmarks\benchmark_tree_walk.py --depth 6 --fanout 2 --out-json results\metrics\single.json

# 扫参（预设本地小网格）
python scripts\benchmarks\sweep_tree_benchmark.py --preset local --out-csv results\metrics\sweep_tree_reader_local.csv

# naive vs fused 对比图（跨机时仅作示意，主文请同机复扫）
python scripts\benchmarks\plot_mamba_naive_vs_fused.py --csv-a results\metrics\sweep_tree_reader_20260410_local5060.csv --label-a "5060 HF naive" --csv-b results\metrics\sweep_autodl_fused.csv --label-b "3090 fused" --out results\metrics\figures\mamba_naive_vs_fused_peak.png
```

---

## 10. 相关文件

| 文件 | 作用 |
|------|------|
| `src/rag_tree/benchmark_core.py` | 可编程入口 `run_tree_reader_benchmark` |
| `scripts/benchmarks/benchmark_tree_walk.py` | 单次 CLI |
| `scripts/benchmarks/sweep_tree_benchmark.py` | 多组 CSV / jsonl 扫参 |
| `experiments/A-20260407-toy-tree-reader-bench/README.md` | 单次实验说明 |
| `scripts/benchmarks/benchmark_text_tree.py` | 文本形叶节点 + 自底向上建树 + 同 reader 基准 |
| `scripts/research/run_path_protocol_cuda.sh` | §7 玩具 S1–S4 CUDA 串行复跑 → `metrics/` |
| `docs/research/RESEARCH_NOTES.md` §7.3.1 | 3090 同边界 ms 对照表（与主文 path-batch 图正交） |
