# 阶段 2 成文草稿（真语料浅树 + 任务）

> **定位**：承接 **`PHASE1_MANUSCRIPT.md`**；**不**替代 **`RESEARCH_STATUS_AND_DIRECTION.md`** 的总览。本节只钉 **阶段 2 叙事边界** 与 **指标列语义**，避免与阶段 1 / §7 / 真 LM 辅线混读。  
> **P1 并入**：核心叙事已迁入 **`PHASE1_MANUSCRIPT.md` §8–§9**（系统+A2-S3、检索头讨论边界）；本文件保留 **表数字、登记修订日志** 与 **`EXPERIMENT_REGISTRY`** 交叉引用。

---

## 1. 阶段 2 要报告什么

- **系统**：在 **Wikitext-2 叶块 → 自底向上平衡树** 上，沿用 **`benchmark_wikitext_tree.py`** 的 **同一 reader 槽位**（TF / GRU / Mamba2 path reader），扩展 **网格**（`num_leaves`、`chunk_len`、`dim` 等见 **`NEXT_RESEARCH_PLAN.md`** 与 **`EXPERIMENT_REGISTRY`** 行 **`A-stage2-wikitext-grid-v1`**）。**5060 CUDA、HF naive Mamba** 的 **墙钟 / m2_peak** 见 **`metrics_result/benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json`**。**3090 fused 同拓扑四格 `dim=128`**（**A2-S2**）见 **`benchmark_wikitext_stage2_fused_20260409T1035Z_n*_c*.json`** 与 **`benchmark_wikitext_stage2_fused_grid_20260409T1035Z.csv`**。**同 harness `dim=256`** 四格见 **`benchmark_wikitext_stage2_dim256_20260409T1137Z_*`** 与 **`…_grid_20260409T1137Z.csv`**（登记 **A-stage2-wikitext-dim256-v1**）。**大叶数单点**（**32 叶**、**c8**、**dim128**）见 **`benchmark_wikitext_fused_n32_c8_20260409T1140Z.json`**。**叶数扫描**（**`n∈{8,16,32,64}`**、固定 **c8 dim128**）：**`benchmark_wikitext_stage2_leavescale_20260409T1257Z_n{8,16,32,64}_c8.json`** + **`benchmark_wikitext_stage2_leavescale_grid_20260409T1257Z.csv`**（**`TAG=stage2_leavescale`**）；登记 **A-stage2-wikitext-leavescale-v1**。**3090 fused** 下 **Mamba2 峰值** 四叶数约 **53→73→98→152 MiB**（**64 叶** 仍 **百 MiB** 量级，与 **5060 naive GB** 动机对照时分列）。**XL**：**128 / 256 叶** **`benchmark_wikitext_stage2_leavescale_xl_20260409T1322Z_n128_c8.json`**、**`…_20260409T1324Z_n256_c8.json`** + **`benchmark_wikitext_stage2_leavescale_xl_grid_n128_n256_combined.csv`**（**`TAG=stage2_leavescale_xl`**）；登记 **A-stage2-wikitext-leavescale-xl-v1**（**Mamba2 峰值** **≈282 / 562 MiB**，仍 **低于 1 GiB**）。以上与 **5060 naive** **分列**（见 **§3**）。

### 1.1 本地 5060 CUDA：Wikitext 浅树 2×2（**HF naive** Mamba，`dim=128`，`WARMUP=2` `REPS=5`）

| `num_leaves` \\ `chunk_len` | **8** | **12** |
|-----------------------------|------:|-------:|
| **8** | m2_peak **≈1142** MiB | **≈1144** MiB |
| **16** | **≈2248** MiB | **≈2254** MiB |

（表中为 **`mamba2.peak_alloc_mib`**，四份 JSON 同期 **`git_sha`** 以各文件内为准；**TF/GRU** 多在 **数十 MiB**，见原 JSON。）**结论（动机）**：在本机 **naive** 下 **叶数 8→16** 使 Mamba2 峰值 **约翻倍**；**chunk_len 8↔12** 对峰值 **影响很小**（主要随路径 token 总长略变）。**扁平 CSV**（便于贴表）：**`results/metrics_result/benchmark_wikitext_5060_cuda_grid_20260407.csv`**，由 **`scripts/benchmarks/aggregate_wikitext_5060_cuda_grid.py`** 生成。
- **任务（A2-S3）**：在 **同一棵树、同一叶序** 上增加 **至少一个** 可复现的 **效果 proxy**。当前落地脚本：**`scripts/research/task_wikitext_path_pair.py`**。

---

## 2. A2-S3：叶对「同 cohort」二分类

- **标签**：对无序叶对 \((i,j)\)，\(i<j\)，按 **左到右叶索引** 划分块；**\(y=1\)** 当且仅当 \(\lfloor i/b\rfloor=\lfloor j/b\rfloor\)。默认 **`--cohort sibling`** 时 \(b=\texttt{fanout}\)（同父叶对）；**`root_child`** 时 \(b=\texttt{fanout}^{d-1}\)（根下同一子树）。
- **特征**：两叶各走 path reader，取 **池化向量** 后 **拼接** \([z_i,z_j]\)，**岭回归** 二分类；并报告 **raw mean-pool 拼接** 基线。
- **输出 JSON**：`kind: task_wikitext_path_pair`，`ridge_concat.*.test_acc` 等；归档目录惯例 **`results/metrics/`**（与探针脚本一致）。

**与阶段 1 主图的关系**：本任务报告 **准确率类标量**，**不是** path-batch 的 **wall-clock / m2_peak**。正文或表中 **分列**（或分子表），**禁止**与 **`paper_main_*`** 无标注合并。

**已归档 smoke（`split_seed=1` 为例）**：**8 叶** `task_wikitext_path_pair_sibling8_smoke.json`；**16 叶 CPU** `task_wikitext_path_pair_sibling16_cpu.json`；**16 叶 5060 CUDA** `task_wikitext_path_pair_sibling16_cuda5060.json`（与 CPU **指标一致**，`device` 不同）；**root_child** `task_wikitext_path_pair_rootchild16_cpu.json`。主文若只引一条，建议优先 **sibling**（正负更稀疏、较不易被 **mean-pool 基线** 一步分完）。

**已知限制（v0）**：**`root_child`** 在 **16 叶** 上块较大，**确定性 hash 叶嵌入 + 父节点拼接** 可能使 **raw mean 拼接** 已线性可分，读者 **test_acc≈1** 不一定有区分度——后续可换 **更细 cohort**、**heldout 叶**、或 **cloze / 检索** 任务加压。

**叶级 heldout（`--pair-split leaf_heldout --heldout-leaves H`）**：训练叶对仅来自叶索引 **`[0, n-H)`**，测试叶对仅来自 **`[n-H, n)`**，**避免**同一叶同时出现在 train/test 叶对中（仍是一次前向算全体叶嵌入）。归档例：**`task_wikitext_sibling16_leafheldout4_{cpu,cuda5060}.json`**（**6** test 叶对）、**`…_leafheldout6_{cpu,cuda5060}.json`**（**15** test 叶对）、**`…_c12_leafheldout6_*.json`**（**`chunk_len=12`**，与 **c=8** 分列对比）。**test 叶对数**为 **C(H,2)**；**H** 小时 **ridge test_acc** 方差大；主文宜 **较大 H**、**多 seed**，或看 **三 reader 相对排序**。几何标签（块大小、叶对枚举）实现见 **`src/rag_tree/path_pair_geometry.py`**（**`tests/test_path_pair_geometry.py`**）。

**3090 CUDA fused：`init_seed` 扫描（5 种子 × n16 / n32）**：**`results/metrics_result/task_wikitext_sibling{16,32}_c8_leafheldout6_initseed{0..4}_20260409T1438Z.json`**（服务器可写 **`$MAMBA2_RESULTS_ROOT/metrics/`** 或 **`metrics_result/`**，本仓与 **path-batch JSON** 同 hub **`metrics_result/`** 即可）；登记 **A-stage2-wikitext-path-pair-initseed5-3090-v1**；**多种子须用 `--init-seed`**。汇总：**`aggregate_task_wikitext_path_pair_json.py -g 'results/metrics_result/task_wikitext_sibling16_c8_leafheldout6_initseed*_20260409T1438Z.json'`**。

---

## 3. 公平性与机器列（必读）

| 数字来源 | 含义 |
|----------|------|
| **5060 / 本机 CPU** | smoke、脚本验收、**naive** Mamba 或 CPU 读者；可标 **`device=cpu`**。 |
| **5060 CUDA + HF naive** | 与 fused 3090 **不可同表绝对对比**；须标注 **naive**。 |
| **3090 fused** | **`A-20260408-wikitext-3090-fused`** 等登记行；**仅**在列标题或脚注写明 **fused / mamba_ssm**。 |

详见 **`RESEARCH_STATUS_AND_DIRECTION.md` §6**。**A2-S2** 小网格（fused）依赖 **AutoDL** 窗口；**dim128 / dim256 四格** 与 **32 叶单点** 已归档（见上列 JSON）；扩更大网格前主文仍宜 **脚注标 `git_sha` / 机器**。

---

## 4. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-07 | 初稿：**A2-S3** `task_wikitext_path_pair.py` + 列语义 |
| 2026-04-07 | **16 叶** `sibling` / `root_child` JSON 归档；§2 **已知限制**（root_child×16 可能近平凡） |
| 2026-04-07 | **5060 CUDA**：`task_wikitext_path_pair_sibling16_cuda5060.json`；**B-S2+** `probe_path_reader_linear_text16_heldout_cuda5060.json`（与 CPU 岭探针 **分列登记**） |
| 2026-04-07 | **B-S2+ BCE**：`probe_path_reader_linear_text16_heldout_{train50,headonly50}_cuda5060.json`（与 CPU **train50 / headonly50** 对齐） |
| 2026-04-07 | **leaf_heldout H=6**：`task_wikitext_sibling16_leafheldout6_{cpu,cuda5060}.json`（15 test 叶对） |
| 2026-04-07 | **5060 CUDA** Wikitext 2×2：`benchmark_wikitext_5060_cuda_{n8_c8,n8_c12,n16_c8,n16_c12}_20260407.json`；**§1.1** 汇总表 |
| 2026-04-07 | **`benchmark_wikitext_5060_cuda_grid_20260407.csv`**；**`path_pair_geometry`**；**A2-S3** **`chunk_len=12`** leaf_heldout H=6 |
| 2026-04-09 | **A2-S2**：3090 fused 四格 **`STAMP=20260409T1035Z`** 入 **`metrics_result/`**；登记 **A-stage2-wikitext-grid-v1** |
| 2026-04-09 | **A2-S2 R2**：**`TAG=stage2_fused_r2`** **`STAMP=20260409T1110Z`**；**Mamba2 峰值与 R1 一致** |
| 2026-04-09 | **dim256 四格**：**`STAMP=20260409T1137Z`**；登记 **A-stage2-wikitext-dim256-v1**；**§1** 系统 bullet 补链 |
| 2026-04-09 | **32 叶 c8 dim128** 单点；**headcheck** 登记 **X-20260409-wikitext-headcheck** |
| 2026-04-07 | **§1** 系统 bullet：**叶数扫描** 脚本指针；**登记册** 占位 **A-stage2-wikitext-leavescale-v1** |
| 2026-04-09 | **§1**：**`STAMP=20260409T1257Z`** 归档文件名 + **Mamba2 峰值** 一句；与 **登记册** **A-stage2-wikitext-leavescale-v1** 对齐 |
| 2026-04-09 | **§1**：**XL 128/256** **`1322Z`/`1324Z`** + **combined CSV**；**A-stage2-wikitext-leavescale-xl-v1** |
| 2026-04-09 | **§2**：**3090** **A2-S3** **`init_seed`×5**（**n16/n32**、**H=6**）登记 **A-stage2-wikitext-path-pair-initseed5-3090-v1**；**`aggregate_task_wikitext_path_pair_json.py`** |
| 2026-04-09 | **§2**：**`STAMP=20260409T1438Z`** → **`results/metrics_result/`**（**10** JSON） |
