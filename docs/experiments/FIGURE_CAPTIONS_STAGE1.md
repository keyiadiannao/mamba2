# 阶段 1 主图图注（可直接贴正文 / 幻灯片）

> 数据与登记：`EXPERIMENT_REGISTRY` **A-20260408-paper-main-3090-***、**-pair**；CSV 示例路径 `results/metrics_result/paper_main_*_paper_main_{v1,naive_v1}.csv`。

---

## P0 叙事边界（中文摘要）

**主文主图（naive vs fused 峰值 + Wikitext 同 harness）**回答的是：在**固定建树与 path reader 协议**下，对一批根—叶路径做 **path-batch 前向**时的计时与 **Mamba2 峰值显存**（`max_memory_allocated`）。该 harness **不实现**树上的 DFS 试错序，也**不**把全模型 KV 分项摊进同一张主图。

**§7.2–§7.3 与附录表 §7.3.1** 属于**另一套可复现玩具协议**：在**单条**合成路径或专用 Transformer trunk 上，**分别**测量 S1（Mamba `DynamicCache` clone）、S4（restore）、S2（TF-R1 无 KV 整段前向）、S3（TF-KV 增量）等；**各列物理含义不同**，**不得**与主图曲线混为同一「一步」或互相做差得出结论。

**SSGS × Mamba（`dfs_ssgs_mamba`）** 再占一条线：**按 token 前向 + `DynamicCache` + DFS 回溯**，用于证明**导航环**与迹的一致性（登记 **X-20260421-ssgs-mamba-dfs-demo**）。它与 path-batch 扫参、与 §7.3.1 各列**仍非同一实验**。

**正文可粘贴的一句边界**：主文图呈现 **path-batch 系统级曲线**；§7 表呈现 **分解尺上的玩具对照**；SSGS demo 呈现 **状态快照式 DFS 的可行性**，三者口径须在段落中显式区分。

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
