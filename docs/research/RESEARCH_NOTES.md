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

- 当前 **`Mamba2PathReader` + `benchmark_*`**：验证 **单路径 / 批路径**上的延迟与显存。
- **SSGS**：`src/rag_tree/ssgs.py` — `dfs_ssgs`（id 状态）、`dfs_ssgs_tensor`（占位向量 h）、**`MambaNavState` + `dfs_ssgs_mamba`**（HF **`Mamba2Model` + `DynamicCache`**，按 **token** 前向 + **clone / zero_ / copy_** 快照，与 §7 S1/S4 同张量族）；单测 `tests/test_ssgs.py`、`tests/test_ssgs_mamba.py`；演示 `scripts/research/demo_ssgs_mamba_dfs.py`。**可复现登记**：`EXPERIMENT_REGISTRY` **X-20260421-ssgs-mamba-dfs-demo**（CPU/CUDA 同 DFS 迹；CUDA 用 **`torch_forward`** 补丁）。
- **仍非「全 LM」**：未接生成头、长上下文任务 loss；回溯玩具数见 §7.3.1 与 **X-20260421-***。

---

## 7. 测量协议草案（P1，可写进实验章节）

### 7.0 阶段 1 已报告内容（与 §7.2–7.3 的边界）

**中文摘要（P0，叙事边界）**：阶段 1 **主文素材**来自 **path-batch** 三 reader 基准（含 naive/fused 峰值图与 Wikitext 浅树），度量的是**沿路径批量前向**的耗时与显存峰值，**不是**树上 DFS 试错、**也不是**全 LM KV 分项。**§7.2–§7.3** 与 **§7.3.1** 表为**独立玩具协议**（S1/S2/S3/S4 等），各列计时对象不同，**禁止**与主图曲线混读为同一物理「一步」。**SSGS×Mamba**（§6，`dfs_ssgs_mamba`）是第三条线：token 步进 + cache 快照下的 **DFS 导航环**，登记见 **X-20260421-ssgs-mamba-dfs-demo**。详细图注与表注句稿见 **`docs/experiments/FIGURE_CAPTIONS_STAGE1.md`** 篇首「P0 叙事边界」。

当前仓库中**已写入论文素材、且与下文「回溯协议」区分开**的部分如下：

| 对象 | 说明 |
|------|------|
| **任务** | **平衡玩具树**上，对同一批根—叶路径批量跑三个 **path reader**：`TransformerPathReader`、`GRUPathReader`、`Mamba2PathReader`（HF `Mamba2Model` + `inputs_embeds`）。入口：`run_tree_reader_benchmark` / `sweep_tree_benchmark.py`；语料型树为 `benchmark_text_tree.py` / `benchmark_wikitext_tree.py`（同一 reader 槽位）。 |
| **CSV 字段** | 每次运行记录 `tf_*` / `gru_*` / `m2_*` 的耗时与 **`m2_peak_mib` 等**：对 Mamba2 路径为 **`torch.cuda.max_memory_allocated()` 在单次 reader 基准内的峰值增量**（与脚本 `benchmark_reader` 一致）；**不是** KV cache 分项；§7.2 **TF-R1 / TF-KV** 的玩具测量在独立脚本 **`benchmark_tf_r1_path_segments.py`** / **`benchmark_tf_kv_path_segments.py`**，**未**并入本扫参 harness。 |
| **naive vs fused** | 由环境是否可 import **`mamba_ssm` / `causal_conv1d`** 决定 HF 是否走融合内核；**同机**成对数据见 `EXPERIMENT_REGISTRY` **A-20260408-paper-main-3090-fused** / **-naive** / **-pair**，主图 `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`。 |
| **尚未接线** | **全规模 LLM** 与**生产级可学习导航**（大规模训练、与 SSGS 同轨）仍在扩展；**玩具**侧已有 **最小因果 LM 闭环**（**X-20260422**）+ **启发式导航指标**（**X-20260423**）+ **目标叶条件子头**（**X-20260424**，见 §7.4）。§7.3 玩具 **S1–S4** 已分脚本登记 **X-20260421-***；**SSGS** 已可跑 **Mamba `DynamicCache`** 版 DFS（§6）。**不与** path-batch 扫参 CSV harness 混为一谈。 |

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
- **指标表**：至少一行 **「同等树深、同等试错次数」** 下 **SSM（快照 clone + restore）** 与 **TF-R1**、**TF-KV** 对比（玩具 JSON 见下节草稿）。

**玩具协议 JSON（同路径 depth4 / chunk8 / dim128，3090 CUDA，`6fa7873`）**：S1 `mamba2_cache_snap_segments_depth4_cuda_20260421.json`（`clone_wall_ms`）；S2 `tf_r1_path_segments_depth4_cuda_20260421.json`（`forward_mean_ms`）；S3 `tf_kv_path_segments_depth4_cuda_20260421.json`（`increment_last_chunk_mean_ms` / `kv_cache_nbytes`）；S4 `mamba2_cache_restore_depth4_cuda_same_20260421.json` 与 **`mamba2_cache_restore_depth4_cuda_fromcpu_20260421.json`**（`restore_wall_ms`）。

**复跑**：`bash scripts/research/run_path_protocol_cuda.sh`（输出带 UTC 时间戳文件名；未设 `MAMBA2_RESULTS_ROOT` 时写入 `results/metrics/`）。

#### 7.3.1 玩具 3090 对照（每边界一行，ms）

登记 **X-20260421-*** 系列；**口径不同**（clone / restore / 整段 TF / 增量 KV），勿混为同一物理「一步」。

| seg | seq | S1 clone | S4 restore same | S4 restore from-CPU | S2 TF-R1 fwd | S3 TF-KV inc chunk |
|-----|-----|----------|-----------------|---------------------|--------------|---------------------|
| 0 | 8 | 0.159 | 0.051 | 0.130 | 0.544 | 1.99 |
| 1 | 16 | 0.134 | 0.068 | 0.170 | 0.559 | 4.05 |
| 2 | 24 | 0.109 | 0.052 | 0.139 | 0.573 | 6.08 |
| 3 | 32 | 0.076 | 0.054 | 0.133 | 0.570 | 8.03 |
| 4 | 40 | 0.075 | 0.055 | 0.130 | 0.583 | 10.1 |

（表中 S1/S4/S2/S3 数值来自 `results/metrics/*_20260421.json`，四舍五入；S3 为 **KV 路径上多跑一个 chunk**，非整段重算。）

### 7.4 与当前代码的映射

- **结构 / 试错顺序**：`ssgs.dfs_ssgs`（id 列表）；`ssgs.dfs_ssgs_tensor` + `TensorNavState`（**h∈ℝ^D** 上 `clone`/`copy_`，单测 `tests/test_ssgs.py`）。
- **纯快照带宽下界**：`scripts/benchmarks/benchmark_ssgs_tensor_overhead.py`（导航 wall-clock + 50k 次 clone+restore 均摊）；固定配置 JSON 见 `EXPERIMENT_REGISTRY` **X-20260421-ssgs-tensor-overhead-fixed**。
- **TF-R1（重算）与 S1 对齐的玩具协议**：`scripts/research/benchmark_tf_r1_path_segments.py` — 单路径、累积前缀、`TransformerPathReader` **仅前向**、无 KV；每边界输出 `forward_mean_ms` 与（CUDA）`peak_alloc_mib`。与 `benchmark_tree_walk` 中带 backward 的 `benchmark_reader` **不是**同一计时定义。
- **TF-KV（截断 + 续写）玩具协议**：`scripts/research/benchmark_tf_kv_path_segments.py` — Pre-LN 因果 trunk、**每层 MHA 的 K/V 缓存**；与 S1 同单路径上报 `kv_cache_nbytes` 与「仅前向最后一节点 chunk」的 `increment_last_chunk_mean_ms`；`--branch-truncate-demo` 复现根下错子 chunk → `truncate_kv` → 兄弟 chunk（KV 字节与截断 ms）。
- **SSM restore（§7.3）**：`scripts/research/benchmark_mamba2_cache_restore_segments.py` — S1 同款累积前向得到 `DynamicCache` 后，`clone` 快照；每 rep：`zero_` 活动张量再 `copy_` 还原；`restore_wall_ms`；`--snapshot-device cpu` 时含 CPU→GPU。
- **SSGS × Mamba（导航环）**：`src/rag_tree/ssgs.py` 中 **`MambaNavState` / `dfs_ssgs_mamba` / `build_toy_mamba2_for_ssgs`**；`mamba_cache_utils.patch_mamba2_model_use_torch_forward_only` 在 **CUDA** 上强制 **HF ``torch_forward``**（避免 fused ``causal_conv1d`` 在 **batch=1** 下的 stride 限制）；cache **clone/restore** 同文件。与 path reader **不同**：**DFS 试错序** + **token 步进**（非单次整段 `inputs_embeds`）。**可归档 JSON**：`scripts/research/demo_ssgs_mamba_dfs.py --out-json` → `results/metrics/ssgs_mamba_dfs_demo_{cpu,cuda}_20260421.json`（登记 **X-20260421-ssgs-mamba-dfs-demo**）。
- **真实因果 LM（最小闭环）**：`src/rag_tree/tree_lm_closure.py` — 根—叶路径上节点 ``text`` 拼文档 → HF **`AutoModelForCausalLM`** 的 teacher-forcing **CE** + **`generate` 续写**；`train_one_step_mean_loss` 提供 **一步 AdamW**。入口：`scripts/research/demo_tree_lm_minimal.py`（默认 ``sshleifer/tiny-gpt2``）。**登记**：**X-20260422-tree-lm-minimal**。**不是** Mamba path-reader 基准；**启发式导航指标** 见下行。
- **树上导航（启发式任务指标）**：`src/rag_tree/tree_lm_nav_eval.py` — 内部节点对每个子算「walk+子」整段文档的 **CE**，**argmin** 贪心下降；与金叶路径逐步对比得 **`child_choice_accuracy`**、是否 **`reached_target_leaf`**；走错枝后 **gold_child=-1**（仅 **对象同一性** 对齐金路径，勿用 ``TreeNode`` 值比较）。入口：`scripts/research/demo_tree_lm_nav_greedy.py`；**`--eval-all-leaves`** 得 **`reach_rate`**。登记 **X-20260423-tree-lm-nav-greedy**；归档 **`results/metrics/tree_lm_nav_greedy_default8_cpu.json`** 与 **`tree_lm_nav_greedy_default8_cuda.json`**（CPU/CUDA **同指标**）。加载 `tiny-gpt2` 时 Hub 可能打印 **UNEXPECTED** 键与 **`loss_type=None`** 提示，可忽略。**非**训练策略，可作弱基线或训练后对照。
- **树上导航（目标叶条件、可学习子指针）**：`src/rag_tree/tree_lm_nav_learned.py` — 冻结 **tiny-gpt2**，**最后一 token 隐状态** 与 **`goal_leaf_index` 嵌入** 拼接后线性层 → 各子 **logits**；监督 **CrossEntropy**（默认 8 叶深度 3 二元树 **24** 条内部节点样本）；训练为 **每 epoch 打乱后全量累积梯度、一步 AdamW**。推理时对指定目标叶贪心下降，指标与上行相同。入口：`scripts/research/demo_tree_lm_nav_learned.py`。登记 **X-20260424-tree-lm-nav-learned**；归档 **`tree_lm_nav_learned_default8_cpu.json`** 与 **`tree_lm_nav_learned_default8_cuda.json`**（**CPU/CUDA** 上 **reach_rate** / **mean_child_choice_accuracy** 与 **per_leaf** 模式一致，与 **X-20260423** 同）。与 **X-20260423** 的边界：**必须传入目标叶**（同一前缀在不同目标下金孩子不同）；**非**盲导航。

### 7.5 接线顺序（定稿：先做什么、后做什么）

| 步骤 | 内容 | 状态（截至本仓库当前迭代） |
|------|------|---------------------------|
| S0 | 阶段 1 path reader 扫参 + **同机** naive/fused 峰值图 + Wikitext 同 harness | **已完成**（§7.0 / `FIGURE_CAPTIONS_STAGE1.md`） |
| S1 | **SSM 快照对象**在代码中从玩具 `TensorNavState` 对齐到 **HF `Mamba2Model` 可导出状态**（或正文声明的等价张量） | **已完成（玩具协议）**：探针 + §7.1 表；**累积前缀 + clone cache**：`benchmark_mamba2_cache_snapshot_segments.py`；登记 **X-20260421-mamba2-cache-segments-{cpu,cuda}** |
| S2 | **TF-R1**：同一棵树、固定试错序列下，实现「回退 → 从规定起点重算」的 wall-clock + 峰值 | **已完成（玩具协议）**：`benchmark_tf_r1_path_segments.py`；3090 JSON 与登记 **X-20260421-tf-r1-path-segments-cuda** |
| S3 | **TF-KV**：同一协议下「截断子分支 KV → 续算兄弟分支」+ KV 字节统计 | **已完成（玩具协议）**：`benchmark_tf_kv_path_segments.py`；登记 **X-20260421-tf-kv-path-segments-cuda**；JSON：`tf_kv_path_segments_depth4_cuda_20260421.json` + **`tf_kv_path_segments_depth4_cuda_branchdemo_20260421.json`**（`branch_truncate_demo` / 截断 ms） |
| S4 | **SSM restore**：与 §7.3 一致，仅测 `clone`/`copy_` 或 `load_state` 的 **restore_wall_ms**（可与 S1 后真实张量尺寸一起报） | **已完成（玩具协议）**：`benchmark_mamba2_cache_restore_segments.py`；登记 **X-20260421-mamba2-cache-restore-cuda**（same + fromcpu 两 JSON） |
| S5 | 汇总表：**同等树深、同等试错次数** 下 SSM vs TF-R1 vs TF-KV（或两列 TF） | **部分**：§7.3.1 表 + JSON；**导航环**已接 **`dfs_ssgs_mamba`**；**真 LM**：**X-20260422** + **启发式 reach_rate**（**X-20260423**）+ **目标叶条件可学习子头**（**X-20260424**，默认 8 叶上 **reach_rate** 高于 **X-20260423**）；**与 SSGS 同轨 / 全拟合** 仍可扩展 |

**依赖关系**：S2/S3 依赖清晰的 **token 边界** 与 **路径枚举**（可与 `dfs_ssgs` 轨迹对齐）；S1 完成前，勿把玩具 `dim` 向量与「层状态字节数」混称为论文主表。

