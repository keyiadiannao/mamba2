# 阶段 2 成文草稿（真语料浅树 + 任务）

> **定位**：承接 **`PHASE1_MANUSCRIPT.md`**；**不**替代 **`RESEARCH_STATUS_AND_DIRECTION.md`** 的总览。本节只钉 **阶段 2 叙事边界** 与 **指标列语义**，避免与阶段 1 / §7 / 真 LM 辅线混读。

---

## 1. 阶段 2 要报告什么

- **系统**：在 **Wikitext-2 叶块 → 自底向上平衡树** 上，沿用 **`benchmark_wikitext_tree.py`** 的 **同一 reader 槽位**（TF / GRU / Mamba2 path reader），扩展 **网格**（`num_leaves`、`chunk_len`、`dim` 等见 **`NEXT_RESEARCH_PLAN.md`** 与 **`EXPERIMENT_REGISTRY`** 行 **`A-stage2-wikitext-grid-v1`**）。**5060 CUDA、HF naive Mamba** 的 **墙钟 / m2_peak** 动机样例见 **`results/metrics_result/benchmark_wikitext_5060_cuda_n8_c12_20260407.json`**、**`…_n16_c12_20260407.json`**（与 **3090 fused** **分列**，见 **§3**）。
- **任务（A2-S3）**：在 **同一棵树、同一叶序** 上增加 **至少一个** 可复现的 **效果 proxy**。当前落地脚本：**`scripts/research/task_wikitext_path_pair.py`**。

---

## 2. A2-S3：叶对「同 cohort」二分类

- **标签**：对无序叶对 \((i,j)\)，\(i<j\)，按 **左到右叶索引** 划分块；**\(y=1\)** 当且仅当 \(\lfloor i/b\rfloor=\lfloor j/b\rfloor\)。默认 **`--cohort sibling`** 时 \(b=\texttt{fanout}\)（同父叶对）；**`root_child`** 时 \(b=\texttt{fanout}^{d-1}\)（根下同一子树）。
- **特征**：两叶各走 path reader，取 **池化向量** 后 **拼接** \([z_i,z_j]\)，**岭回归** 二分类；并报告 **raw mean-pool 拼接** 基线。
- **输出 JSON**：`kind: task_wikitext_path_pair`，`ridge_concat.*.test_acc` 等；归档目录惯例 **`results/metrics/`**（与探针脚本一致）。

**与阶段 1 主图的关系**：本任务报告 **准确率类标量**，**不是** path-batch 的 **wall-clock / m2_peak**。正文或表中 **分列**（或分子表），**禁止**与 **`paper_main_*`** 无标注合并。

**已归档 smoke（`split_seed=1` 为例）**：**8 叶** `task_wikitext_path_pair_sibling8_smoke.json`；**16 叶 CPU** `task_wikitext_path_pair_sibling16_cpu.json`；**16 叶 5060 CUDA** `task_wikitext_path_pair_sibling16_cuda5060.json`（与 CPU **指标一致**，`device` 不同）；**root_child** `task_wikitext_path_pair_rootchild16_cpu.json`。主文若只引一条，建议优先 **sibling**（正负更稀疏、较不易被 **mean-pool 基线** 一步分完）。

**已知限制（v0）**：**`root_child`** 在 **16 叶** 上块较大，**确定性 hash 叶嵌入 + 父节点拼接** 可能使 **raw mean 拼接** 已线性可分，读者 **test_acc≈1** 不一定有区分度——后续可换 **更细 cohort**、**heldout 叶**、或 **cloze / 检索** 任务加压。

**叶级 heldout（`--pair-split leaf_heldout --heldout-leaves H`）**：训练叶对仅来自叶索引 **`[0, n-H)`**，测试叶对仅来自 **`[n-H, n)`**，**避免**同一叶同时出现在 train/test 叶对中（仍是一次前向算全体叶嵌入）。归档例：**`task_wikitext_sibling16_leafheldout4_{cpu,cuda5060}.json`**（**6** test 叶对）、**`…_leafheldout6_*.json`**（**15** test 叶对，**CPU/CUDA** 更同向）。**test 叶对数**为 **C(H,2)**；**H** 小时 **ridge test_acc** 方差大；主文宜 **较大 H**、**多 seed**，或看 **三 reader 相对排序**。

---

## 3. 公平性与机器列（必读）

| 数字来源 | 含义 |
|----------|------|
| **5060 / 本机 CPU** | smoke、脚本验收、**naive** Mamba 或 CPU 读者；可标 **`device=cpu`**。 |
| **5060 CUDA + HF naive** | 与 fused 3090 **不可同表绝对对比**；须标注 **naive**。 |
| **3090 fused** | **`A-20260408-wikitext-3090-fused`** 等登记行；**仅**在列标题或脚注写明 **fused / mamba_ssm**。 |

详见 **`RESEARCH_STATUS_AND_DIRECTION.md` §6**。**A2-S2** 小网格（fused）仍依赖 **AutoDL** 窗口；未跑前不占位主文硬数字。

---

## 4. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-07 | 初稿：**A2-S3** `task_wikitext_path_pair.py` + 列语义 |
| 2026-04-07 | **16 叶** `sibling` / `root_child` JSON 归档；§2 **已知限制**（root_child×16 可能近平凡） |
| 2026-04-07 | **5060 CUDA**：`task_wikitext_path_pair_sibling16_cuda5060.json`；**B-S2+** `probe_path_reader_linear_text16_heldout_cuda5060.json`（与 CPU 岭探针 **分列登记**） |
| 2026-04-07 | **B-S2+ BCE**：`probe_path_reader_linear_text16_heldout_{train50,headonly50}_cuda5060.json`（与 CPU **train50 / headonly50** 对齐） |
| 2026-04-07 | **leaf_heldout H=6**：`task_wikitext_sibling16_leafheldout6_{cpu,cuda5060}.json`（15 test 叶对） |
| 2026-04-07 | **5060 CUDA** Wikitext 计时：`benchmark_wikitext_5060_cuda_{n8_c12,n16_c12}_20260407.json`（**metrics_result/**） |
