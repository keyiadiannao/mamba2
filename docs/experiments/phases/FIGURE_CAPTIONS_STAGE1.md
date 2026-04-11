# 阶段 1 主图图注（可直接贴正文 / 幻灯片）

> 数据与登记：`EXPERIMENT_REGISTRY` **A-20260408-paper-main-3090-***、**-pair**；CSV 示例路径 `results/metrics_result/paper_main_*_paper_main_{v1,naive_v1}.csv`。

---

## P0 叙事边界（中文摘要）

**主文主图（naive vs fused 峰值 + Wikitext 同 harness）**回答的是：在**固定建树与 path reader 协议**下，对一批根—叶路径做 **path-batch 前向**时的计时与 **Mamba2 峰值显存**（`max_memory_allocated`）。该 harness **不实现**树上的 DFS 试错序，也**不**把全模型 KV 分项摊进同一张主图。

**§7.2–§7.3 与附录表 §7.3.1** 属于**另一套可复现玩具协议**：在**单条**合成路径或专用 Transformer trunk 上，**分别**测量 S1（Mamba `DynamicCache` clone）、S4（restore）、S2（TF-R1 无 KV 整段前向）、S3（TF-KV 增量）等；**各列物理含义不同**，**不得**与主图曲线混为同一「一步」或互相做差得出结论。

**SSGS × Mamba（`dfs_ssgs_mamba`）** 再占一条线：**按 token 前向 + `DynamicCache` + DFS 回溯**，用于证明**导航环**与迹的一致性。**玩具树** 登记 **X-20260421-ssgs-mamba-dfs-demo**；**与 `benchmark_wikitext_tree` 同建树** 的 **Wikitext** 归档见 **X-20260407-ssgs-mamba-wikitext-tree**（**`results/metrics_result/ssgs_mamba_wikitext_grid.csv`**：通配合并 **多 JSON**，本仓 **16 行** 量级（**`aggregate` 合并后**；以 **`ssgs_mamba_wikitext_grid.csv`** 为准）；列 **`snapshots_taken` / `rollbacks` / `leaf_checks`**，**非** path-batch **wall-clock**；**`json_path`** 若来自服务器，脚注写 **basename**）。**辅**：同树 **path-batch 三 reader** 小 smoke **`benchmark_wikitext_ssgs_bundle_20260410T0803Z_n8_c8.json`**（**与 SSGS 计数分列**）。它与 path-batch 全网格扫参、与 §7.3.1 各列**仍非同一实验**。

**Phase M1（SSGS vs 玩具 TF-KV，同树 DFS）** 再占一条线：与 **`demo_ssgs_mamba_wikitext` 同 Wikitext 建树**、**同一 DFS 目标叶** 上，**Mamba 臂** 与 **IncrementalCausalTransformerKV** 的 **full KV clone/restore**、**truncate_kv** 两臂 **并表**（**`kind=ssgs_vs_kv_tree_nav_wikitext`**）；报告 **wall_s / peak / KV 字节或截断次数** 等。该 **玩具 TF-KV trunk** **不是** **`TransformerPathReader`**，也**不是** path-batch **三 reader** 槽位。可选 **L3**：末 token hidden **余弦**（vs 金路径-only）、**固定随机叶头 CE**（**`abs_ce_delta`**）；**与树 LM** **X-20260423/24**（CE 路由 vs 可学习子头）**不同 harness**。登记 **X-ssgs-vs-kv-tree-nav-m1**；汇总 **`ssgs_vs_kv_wikitext_nav_grid.csv`**（通配 **`ssgs_vs_kv_tree_nav_wikitext_*.json`**）。

**L3 轨迹甲·乙（玩具 TF-KV，硬编码读序）**：与 **M1 全树 DFS** **不同** **`kind`**（**`tf_kv_trajectory_l3_minimal`**）：在 **小平衡树**（默认 **depth=2, fanout=2**）上比较 **错枝一步 + restore + 金路径后缀** 与 **金路径直达** 的 **末 token hidden 余弦**（**clone** / **truncate_kv** 两臂各报）。见 **`RESEARCH_STATUS` §3.5**、**`benchmark_tf_kv_trajectory_l3_minimal.py`**；登记 **X-20260411-tf-kv-trajectory-l3-minimal**。

**正文可粘贴的一句边界**：主文图呈现 **path-batch 系统级曲线**；§7 表呈现 **分解尺上的玩具对照**；SSGS demo 呈现 **状态快照式 DFS 计数可行性**；**M1** 呈现 **同树 DFS 上 SSGS 与玩具 TF-KV 的代价并表（含可选 L3）**；**轨迹 L3** 呈现 **受控错枝/恢复 vs 直达** 的 **表示一致性**——**各 harness 须在段落中显式区分**。

---

## 七条测量轴（与 `RESEARCH_STATUS_AND_DIRECTION.md` §3 一致；防混读）

正文 **禁止**把下列 **不同轴** 的纵轴或列 **当作同一物理量** 相减、同图并列无标注、或一句里混谈。

| 轴 | 回答什么 | 代表登记 / 文件 |
|----|----------|-----------------|
| **Path-batch 主图** | 固定路径集合上 **三 reader 批量前向** 的 **时间与 m2_peak** | **A-20260408-paper-main-3090-pair**；`paper_main_*.csv` |
| **§7 玩具表** | **单路径**上 **clone / restore / TF-R1 / TF-KV** 等 **分列毫秒** | **X-20260421-***；`*_20260421.json` |
| **SSGS demo** | **DFS 试错序** + **token 步进** + cache 快照（**计数**：snapshots / rollbacks） | **X-20260421**（玩具）；**X-20260407** + **`ssgs_mamba_wikitext_grid.csv`**（Wikitext 同树；**多 STAMP** 合并，**16 行** 量级） |
| **M1 同树 DFS 对照** | **同 Wikitext 建树**上 **SSGS Mamba** vs **玩具 TF-KV**（clone / **truncate_kv**）**同一 DFS**；**wall_s / peak / KV**；可选 **L3**（隐状态、固定叶头 CE） | **X-ssgs-vs-kv-tree-nav-m1**；**`ssgs_vs_kv_tree_nav_wikitext_*.json`**；**`ssgs_vs_kv_wikitext_nav_grid.csv`** |
| **L3 轨迹（玩具 TF-KV）** | **硬编码** 错枝一步 **+ restore** **+** 金后缀 **vs** 金路径直达；**末 hidden 余弦 / 墙钟**；**≠ M1 DFS** | **X-20260411-tf-kv-trajectory-l3-minimal**；**`tf_kv_trajectory_l3_minimal_*.json`** |
| **真 LM 线** | **tiny-gpt2** 上 **CE / 导航指标**；**非** path-batch harness | **X-20260422–25** |
| **阶段 2 任务（A2-S3）** | 同 Wikitext 树上的 **效果 proxy**（例：叶对 cohort **ridge 准确率**）；**非**主图纵轴、**非** §7 毫秒 | **A-20260407-stage2-wikitext-path-pair**；**`task_wikitext_*.json`**；贴表优先 **`task_wikitext_sibling*_initseed5_summary_20260410T*.tsv`**（见 **`SUBMISSION_PACK` §A2**） |

**阶段 2 效率补充（5060 Wikitext 2×2）**：**不是** 主图上的新曲线，而是 **本地动机表**（**HF naive**、`m2_peak_mib`）；数据 **`benchmark_wikitext_5060_cuda_grid_20260407.csv`** 与四份 JSON；**禁止**与 **3090 fused** 主文格点 **无脚注同表**。若正文出现「5060 + A2-S3」同段，须 **分列**（左：效率/m2_peak；右：任务 test_acc）或 **分子表**，并各注 **device / naive·fused / split**。

**与 `PHASE1_MANUSCRIPT.md` 的对照**：阶段 2 **成文素材**见 **`PHASE1_MANUSCRIPT.md` §8**；**检索头 / Mamba 不可机械对应** 见 **§9** 与 **`docs/research/RETRIEVAL_HEAD_NOTES.md` §8**；**七轴** 与 **`SUBMISSION_PACK` §A3** 一致。

---

**真 LM 玩具导航（登记 X-20260422–24）**：路径文本上的 **teacher-forcing / 子文档 CE / 目标叶条件可学习子头**，**都不是**主图 path-batch，也**不是** SSGS 里 **Mamba `DynamicCache` + token 步进 DFS** 的同一条实验。**X-20260424** 在默认 8 叶、冻结 **tiny-gpt2** 与浅线性头下 **reach_rate 仍 &lt; 1**（登记约 **0.375**），正文**不应**写成「已解决全树导航」；要强宣称需另给架构、数据或训练设定。**与 SSGS 的并列对照**（同文本树、按 goal 统计 DFS 代价 vs 子头贪心）见 `scripts/research/demo_ssgs_lm_nav_compare.py`，登记 **X-20260425-ssgs-lm-nav-compare**。

---

## 图：`mamba_3090_naive_vs_fused_dim128_paper_main_v1.png`

**建议英文图注（短）**

> Mamba2 peak GPU memory (`torch.cuda.max_memory_allocated`, MiB) on the same **preset-local** grid (8 configurations: `num_leaves` × `chunk_len`, `fanout=2`, `dim=128`), **RTX 3090**, **same code revision** (`6fa7873`), **WARMUP=2**, **REPS=8**. **Naive**: HuggingFace `Mamba2PathReader` without `mamba_ssm` / `causal_conv1d` (conda env `mamba2_naive`). **Fused**: same script with fused kernels installed. Curves: `plot_mamba_naive_vs_fused.py`.

**建议中文图注（短）**

> 在 **RTX 3090**、**同一提交 `6fa7873`**、**WARMUP=2、REPS=8** 下，玩具树 **dim=128、preset local** 八个格点上的 Mamba2 **峰值显存**（`max_memory_allocated`，MiB）。**Naive**：无 `mamba_ssm`/`causal_conv1d` 的 HF 回退；**Fused**：安装融合栈后的同脚本结果。

---

## 图：`mamba_3090_naive_vs_fused_dim256_paper_main_v1.png`

**英文（短）**

> Same as dim128 figure, **dim=256**, **12** grid points (`depth` 3–6 × `chunk_len` ∈ {4,8,12}, `fanout=2`, `max_leaves=64`), same machine, warmup/reps, and naive/fused definitions.

**中文（短）**

> 同机同设定，**dim=256**，**12** 个格点（深度 3–6 × chunk 4/8/12，`fanout=2`），naive / fused 定义同上。

---

## 图：`mamba_3090_naive_vs_fused_dim384_paper_main_v1.png`

**英文（短）**

> Same as above, **dim=384**, **6** grid points (`depth` 4–6 × `chunk_len` ∈ {8,16}, `fanout=2`, `max_leaves=64`).

**中文（短）**

> 同机同设定，**dim=384**，**6** 个格点（深度 4–6 × chunk 8/16）。

---

## Wikitext 浅树（3090 fused）

登记：**A-20260408-wikitext-3090-fused**；JSON：`results/metrics_result/benchmark_wikitext_3090_fused_20260408T0846Z.json`。

**一句说明（可放正文）**

> 在 **Wikitext-2 raw** 叶块构造的浅树上，使用与合成树相同的 **TF / GRU / Mamba2 path reader** 槽位；**AutoDL RTX 3090**、**fused** 环境；Hub 经 **`HF_ENDPOINT` 镜像**拉取数据（见 `AUTODL_SETUP` §2b）。本地 5060 对照见 **A-20260410-wikitext-shallow-tree**。

---

## 与 §7 回溯实验的边界

上述三张 naive/fused 图与 Wikitext 说明仍只服务 **单遍前向 path-batch** 峰值显存与计时。

**玩具级回溯协议**（HF Mamba `DynamicCache` + 手写 TF-KV trunk）已单独跑数并登记 **EXPERIMENT_REGISTRY `X-20260421-*`**；**每边界毫秒对照表**见 **`RESEARCH_NOTES.md` §7.3.1**，原始 JSON 见 `results/metrics/*_20260421.json`。复跑：`bash scripts/research/run_path_protocol_cuda.sh`。正文若引用，请标明 **toy / 非全 LLM KV** 与 §7.0 边界。

---

## 附录表（非主图）：§7.3.1 玩具回溯协议对照

**放哪里**：建议 **附录** 或 **补充材料**；**不要**与 naive/fused 主图并列为 Figure 1 的并列子图，以免读者误以为同一实验 harness。

**英文图注（表注，长）**

> **Auxiliary toy protocol (not the path-batch reader sweep).** Per-tree-boundary timings on a **single root-to-leaf synthetic path** (`depth=4`, `chunk_len=8`, `dim=128`, `fanout=2`), **RTX 3090**, **commit `6fa7873`**. **S1** `clone_wall_ms`: clone HF `Mamba2Model` `DynamicCache` (`conv_states` + `recurrent_states`) after a **full-prefix** forward (`batch=1`). **S4** `restore_wall_ms`: `zero_` live cache then `copy_` from snapshot (**same** GPU vs **CPU-resident** snapshot). **S2** `forward_mean_ms`: one **full** `TransformerPathReader` forward on the cumulative prefix (**TF-R1**, no KV). **S3** `increment_last_chunk_mean_ms`: **incremental** causal Transformer with **MHA KV cache** (**TF-KV** toy trunk, not the path-reader module). Metrics are **not interchangeable** (different operations per column). Source JSON: `results/metrics/*_20260421.json`; registry ids **X-20260421-***; reproducibility: `scripts/research/run_path_protocol_cuda.sh`. **SSGS navigation** with real Mamba cache is implemented separately (`MambaNavState` / `dfs_ssgs_mamba`, token-wise forward); see `RESEARCH_NOTES` §6–§7.4.

**中文表注（长）**

> **附录级玩具协议（与主文 path-batch 扫参非同一 harness）。** 在**单条**合成根—叶路径上（`depth=4`，`chunk_len=8`，`dim=128`），**RTX 3090**，**`6fa7873`**。**S1** 列为 HF `Mamba2Model` 在**累积前缀**整段前向后的 **`DynamicCache` clone** 耗时；**S4** 列为对活动 cache 做 **`zero_`+`copy_` 还原**（快照在 GPU 或 CPU）；**S2** 为 **TF-R1**（整段 Transformer 前向、无 KV）；**S3** 为 **TF-KV** 玩具 trunk 上**仅新 chunk** 的增量前向。各列**物理含义不同**，不可当作同一「一步」互减。**SSGS** 与 Mamba 真 cache 的接线路径为 **`dfs_ssgs_mamba`**（按 **token** 前向，见 `RESEARCH_NOTES` §6）。数据 JSON：`results/metrics/*_20260421.json`；登记 **X-20260421-***；复跑：`run_path_protocol_cuda.sh`。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | **P0**：篇首增 **Phase M1**；**测量轴** **五→六**（**M1 同树 DFS**）；边界句含 **M1** |
| 2026-04-11 | **P0**：**L3 轨迹甲·乙**（**`tf_kv_trajectory_l3_minimal`**）；**测量轴** **六→七**；与 **M1**、**path-batch** **分列** |
