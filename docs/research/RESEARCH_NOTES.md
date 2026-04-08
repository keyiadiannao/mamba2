# 研究笔记：隐状态快照 vs KV Cache（树导航 / 试错检索）

> 来源：与协作者的讨论整理；**作为工作假说与实验设计备忘**，非已证结论。实现时需与 `docs/overview/PROJECT_MASTER_PLAN.md` §1 中「状态快照 / 廉价回溯」对照阅读。

---

## 1. 核心对比（为何值得做成可测命题）

| 维度 | Transformer + KV Cache | Mamba / SSM（隐状态递推） |
|------|------------------------|---------------------------|
| **记忆载体** | 随上下文长度增长的 K/V（显存与带宽压力常见） | 每层（或块）**固定形状**的状态 \(h_t = f(h_{t-1}, x_t)\) |
| **「走错分支」后的典型代价** | 丢弃 KV 则丢上下文；保留则累积「污染」；回溯常伴随**重算**或**大量缓存管理** | 若在决策点保存 **\(h\)** 快照，回退可视为 **加载固定维向量**，避免为错误分支永久买单 |
| **叙事要点** | 并行友好、长上下文依赖 KV 规模 | 顺序递推友好、**试探性检索**可与「悔棋式」回退结合 |

**需写进实验协议的细节**：真实 Mamba 堆栈有多层 / 多块状态；「快照」应对齐 **哪一层、张量列表如何 clone、与 `cuda` graph 是否冲突**；对照组 Transformer 的 **KV 截断 / 重算从哪一步开始** 必须固定，否则比较不公平。

---

## 2. 概念：状态快照与回溯

- **快照**：在离散时刻 \(t\)（例如读完某个树节点文本之后）保存当前 **SSM 状态**（工程上常为一组张量，而非单一标量向量）。
- **回溯**：丢弃错误子路径上的后续状态更新，将模型状态重置为已保存的快照，再沿另一子树继续递推。
- **与树结构的耦合**：在「父节点 / 决策点」挂快照，子树探索失败则 **O(1) 恢复父状态**（相对「状态张量大小」而言），再试兄弟分支——这是 **State-Snapshot Guided Search（SSGS）** 一类算法的直观描述。

---

## 3. SSGS 算法草图（研究贡献候选名，可改名）

**输入**：根到叶的遍历协议、每节点文本、Mamba reader（或完整 LM 的 SSM 段）。

1. **节点编码**：读完节点 \(v\) 的文本后，得到状态 \(H_v\)（多层则为一组 tensor）。
2. **快照挂载**：`node[v].state_snapshot = clone(H_v)`（仅决策点或每层节点择一策略，需消融）。
3. **探索**：由策略（贪婪 / 采样 / 外加「相关性头」）选择子节点 \(u\)。
4. **错误检测**：启发式或训练好的打分（相关性、任务 loss、外部检索器）。
5. **快速回退**：`H_current ← node[parent_or_checkpoint].state_snapshot`；不保留错误子树上的递推结果。
6. **切换路径**：从同一快照出发处理另一子节点文本。

**论文里需要配的数字**：快照内存 **bytes/节点**、回退 **wall-clock** vs「从根重编码到该深度」、与 **截断 KV + 重算** 基线的对比。

---

## 4. 与 Agent / RAG 的叙事

- **上下文污染**：错误文档填满 prompt 或 KV 时，后续推理变差；快照回退旨在 **撤销错误分支对世界状态的更新**，而非仅在自然语言层面「告诉模型忽略上文」。
- **适用边界**：若系统始终用 **短上下文、单路径、无试错**，优势可能不明显；**多分支探索、深度导航、预算受限** 时叙事更强。

---

## 5. 已知风险（避免过度承诺）

- **短序列 / 小模型**：TF reader 的 KV 体积可能仍小，「廉价回溯」优势需 **长度 × 分支深度** 拉起来才显著。
- **实现成本**：多层状态快照 = 多份 tensor；与「单向量 \(h\)」的科普表述不同，论文与代码应一致。
- **混合与缓存**：业界已有 **KV 压缩、分页、重算策略**；贡献应落在 **同协议、同预算** 下的可复现对比，而非绝对「Transformer 不能做」。

---

## 6. 与仓库现状的关系

- 当前 **`Mamba2PathReader` + `benchmark_*`**：验证 **单路径 / 批路径**上的延迟与显存；**SSGS 最小草稿**见 `src/rag_tree/ssgs.py`（假状态 DFS + `TreeNode.state_snapshot` 挂载钩子），单测 `tests/test_ssgs.py`。
- **下一步工程**：把 `dfs_ssgs` 的「状态」从 `list[id]` 换成 **真实 Mamba 层状态 clone**；与 Transformer **重算基线**对拍计时，再写入 `docs/experiments/EXPERIMENT_REGISTRY.md`。

---

## 7. 测量协议草案（P1，可写进实验章节）

### 7.0 阶段 1 已报告内容（与 §7.2–7.3 的边界）

当前仓库中**已写入论文素材、且与下文「回溯协议」区分开**的部分如下：

| 对象 | 说明 |
|------|------|
| **任务** | **平衡玩具树**上，对同一批根—叶路径批量跑三个 **path reader**：`TransformerPathReader`、`GRUPathReader`、`Mamba2PathReader`（HF `Mamba2Model` + `inputs_embeds`）。入口：`run_tree_reader_benchmark` / `sweep_tree_benchmark.py`；语料型树为 `benchmark_text_tree.py` / `benchmark_wikitext_tree.py`（同一 reader 槽位）。 |
| **CSV 字段** | 每次运行记录 `tf_*` / `gru_*` / `m2_*` 的耗时与 **`m2_peak_mib` 等**：对 Mamba2 路径为 **`torch.cuda.max_memory_allocated()` 在单次 reader 基准内的峰值增量**（与脚本 `benchmark_reader` 一致）；**不是** KV cache 分项；§7.2 **TF-R1 / TF-KV** 的玩具测量在独立脚本 **`benchmark_tf_r1_path_segments.py`** / **`benchmark_tf_kv_path_segments.py`**，**未**并入本扫参 harness。 |
| **naive vs fused** | 由环境是否可 import **`mamba_ssm` / `causal_conv1d`** 决定 HF 是否走融合内核；**同机**成对数据见 `EXPERIMENT_REGISTRY` **A-20260408-paper-main-3090-fused** / **-naive** / **-pair**，主图 `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`。 |
| **尚未接线** | §7.1 **全 LM** 层状态与 SSGS 主循环对接；§7.3 **三方回退表**全文（玩具数字已分脚本：S1/S2/S3 均见 `EXPERIMENT_REGISTRY` **X-20260421-***）；**不与**上述玩具扫参 CSV harness 混为一谈。 |

### 7.1 快照里装什么（Mamba / SSM reader）

| 对象 | 说明 |
|------|------|
| **最小集** | 每一层（或每个 Mamba block）在**读完当前树节点文本、做完该段递推之后**的 **SSM 状态张量**；以 `Module.state_dict()` 子集或 `NamedTuple[torch.Tensor, ...]` 列出，**逐 tensor `clone().detach()`**。 |
| **实现依赖** | HuggingFace `Mamba2Model`：需从 `outputs` / 内部 cache 钩出与 **步进边界** 对齐的状态；若 API 不暴露，则记录「等价物」（例如 block 输入输出拼接前的 hidden）并在正文声明。 |
| **消融** | 仅根快照 / 每层决策点快照 / 全节点快照 —— 三档报告 **bytes/节点** 与 **回退延迟**。 |

**已观测（HF + `Mamba2Model`，`use_cache=True`，Transformers 与仓库脚本一致）**  
命令：`scripts/research/probe_mamba2_outputs.py --use-cache --dump-cache`（CUDA fused 上 batch 自动 ≥8）。

- 前向输出除 `last_hidden_state` 外有 **`cache_params`：`DynamicCache`**，按层为 **`LinearAttentionLayer`**。
- 每层暴露的张量字段（**名称以当前 transformers 为准**）：

**AutoDL / RTX 3090 / fused**（`mamba_ssm`+`causal_conv1d`，与 **X-20260408-probe-mamba2-3090-fused** 一致）：`batch=8`，`seq=32`，`hidden=128`，`num_heads=8`，`head_dim=32`，**2 层**。

| 字段 | 每层形状 | float32 元素数 | 约 nbytes（该张量） |
|------|----------|----------------|---------------------|
| `conv_states` | `(8, 288, 4)` | 9 216 | 36 864 |
| `recurrent_states` | `(8, 8, 32, 16)` | 32 768 | 131 072 |

- **单层**两字段合计约 **167 936 B（≈164 KiB）**；**两层**若全量 `clone` 约 **335 872 B（≈328 KiB）**（不含 `last_hidden_state` 与其它开销）。
- **本机 CPU / naive**（`batch=1` 同脚本）：形状为 `(1, 288, 4)` 与 `(1, 8, 32, 16)`，仅 **batch 维**与上表不同。

**快照候选**：对上述 tensor **逐 `clone().detach()`** 计入 **bytes**；是否与 SSM 论文「层状态」一一对应，正文需**显式声明**并与逐步前向边界对齐（§7.5 S1）。

> 形状随 **`batch`、`seq`、`hidden`、`num_heads`、`head_dim`、`state_size`** 变化；论文数字以本表 + 登记 **X-20260408-probe-mamba2-3090-fused** 为准。

### 7.2 Transformer 对照基线（必须固定其一，勿混用）

| 基线代号 | 做法 | 测量的量 |
|----------|------|----------|
| **TF-R1（重算）** | 回退到父检查点后，**从根（或从该检查点对应的 token 边界）重新前向**到当前分支，**不复用**旧 KV。 | wall-clock、总 FLOPs 估计（可选）、峰值显存。 |
| **TF-KV（截断）** | 保留缓存至父边界，丢弃子分支产生的 KV 后缀，**继续**前向兄弟分支。 | 同上；另报 **KV 占用字节**（按层累加）。 |

**公平性**：两种基线允许的 **token 序列** 必须与 SSGS 路径一致；Mamba 侧 **不得**在回退后仍保留错误子分支的隐状态（除非故意做「污染」消融）。

### 7.3 回退成本怎么报

- **SSM**：`restore_wall_ms = 拷贝快照到设备 + `load_state`（若有）**；不含重新编码子树。
- **TF-R1**：`recompute_wall_ms` = 从固定起点到兄弟叶的全序列前向。
- **指标表**：至少一行 **「同等树深、同等试错次数」** 下三者对比。

### 7.4 与当前代码的映射

- **结构 / 试错顺序**：`ssgs.dfs_ssgs`（id 列表）；`ssgs.dfs_ssgs_tensor` + `TensorNavState`（**h∈ℝ^D** 上 `clone`/`copy_`，单测 `tests/test_ssgs.py`）。
- **纯快照带宽下界**：`scripts/benchmarks/benchmark_ssgs_tensor_overhead.py`（导航 wall-clock + 50k 次 clone+restore 均摊）；固定配置 JSON 见 `EXPERIMENT_REGISTRY` **X-20260421-ssgs-tensor-overhead-fixed**。
- **TF-R1（重算）与 S1 对齐的玩具协议**：`scripts/research/benchmark_tf_r1_path_segments.py` — 单路径、累积前缀、`TransformerPathReader` **仅前向**、无 KV；每边界输出 `forward_mean_ms` 与（CUDA）`peak_alloc_mib`。与 `benchmark_tree_walk` 中带 backward 的 `benchmark_reader` **不是**同一计时定义。
- **TF-KV（截断 + 续写）玩具协议**：`scripts/research/benchmark_tf_kv_path_segments.py` — Pre-LN 因果 trunk、**每层 MHA 的 K/V 缓存**；与 S1 同单路径上报 `kv_cache_nbytes` 与「仅前向最后一节点 chunk」的 `increment_last_chunk_mean_ms`；`--branch-truncate-demo` 复现根下错子 chunk → `truncate_kv` → 兄弟 chunk（KV 字节与截断 ms）。
- **真实 LM**：尚未接线；接线点后应新增独立 benchmark 与 registry id，避免与玩具 `benchmark_tree_walk` 混淆。

### 7.5 接线顺序（定稿：先做什么、后做什么）

| 步骤 | 内容 | 状态（截至本仓库当前迭代） |
|------|------|---------------------------|
| S0 | 阶段 1 path reader 扫参 + **同机** naive/fused 峰值图 + Wikitext 同 harness | **已完成**（§7.0 / `FIGURE_CAPTIONS_STAGE1.md`） |
| S1 | **SSM 快照对象**在代码中从玩具 `TensorNavState` 对齐到 **HF `Mamba2Model` 可导出状态**（或正文声明的等价张量） | **已完成（玩具协议）**：探针 + §7.1 表；**累积前缀 + clone cache**：`benchmark_mamba2_cache_snapshot_segments.py`；登记 **X-20260421-mamba2-cache-segments-{cpu,cuda}** |
| S2 | **TF-R1**：同一棵树、固定试错序列下，实现「回退 → 从规定起点重算」的 wall-clock + 峰值 | **已完成（玩具协议）**：`benchmark_tf_r1_path_segments.py`；3090 JSON 与登记 **X-20260421-tf-r1-path-segments-cuda** |
| S3 | **TF-KV**：同一协议下「截断子分支 KV → 续算兄弟分支」+ KV 字节统计 | **已完成（玩具协议）**：`benchmark_tf_kv_path_segments.py`；3090 JSON 与登记 **X-20260421-tf-kv-path-segments-cuda**；可选 `--branch-truncate-demo` 另报 |
| S4 | **SSM restore**：与 §7.3 一致，仅测 `clone`/`copy_` 或 `load_state` 的 **restore_wall_ms**（可与 S1 后真实张量尺寸一起报） | **部分**：玩具维度的微基准已有 JSON；真实层状态待 S1 |
| S5 | 汇总表：**同等树深、同等试错次数** 下 SSM vs TF-R1 vs TF-KV（或两列 TF） | **未做** |

**依赖关系**：S2/S3 依赖清晰的 **token 边界** 与 **路径枚举**（可与 `dfs_ssgs` 轨迹对齐）；S1 完成前，勿把玩具 `dim` 向量与「层状态字节数」混称为论文主表。

