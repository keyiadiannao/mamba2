# 阶段 1 主图图注（可直接贴正文 / 幻灯片）

> 数据与登记：`EXPERIMENT_REGISTRY` **A-20260408-paper-main-3090-***、**-pair**；CSV 示例路径 `results/metrics_result/paper_main_*_paper_main_{v1,naive_v1}.csv`。

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

上述图与表仅对比 **单遍前向 path-batch** 下的峰值显存与计时；**不包含** `RESEARCH_NOTES` §7.2 的 TF-R1/TF-KV **回溯**协议。见 **§7.0**。
