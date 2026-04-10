# 本机 RTX 5060（Windows）推进手册

> **定位**：与 **3090 / AutoDL** 分列时，在 **本机** 跑 **轻量 / smoke**；产出 **JSON** 入 **`results/metrics/`**（或 **`metrics_result/`**），**登记** 时注明 **5060 / CPU / 本机** 与 **`git_sha`**。  
> **公平性**：**5060 + HF naive** 与 **3090 fused**、**path-batch** 与 **§7** / **SSGS** / **A2-S3** 须 **分列脚注**（**`docs/experiments/phases/FIGURE_CAPTIONS_STAGE1.md`**、**`PHASE1_MANUSCRIPT` §5.1**）。  
> **算力紧张或仅本机时**：优先 **成文（P0）**；本机实验为 **可选**，见 **`docs/overview/execution/NEXT_RESEARCH_PLAN.md`** **「算力不可用时的备选推进」** 小节 **A/B**；**不必**为追新数字重复已完成的 smoke。

---

## 1. 环境（必读）

**勿**使用 **Anaconda base** 里已损坏的 **`c10.dll`**（典型报错：`WinError 1114` / `Error loading c10.dll`）。请使用 **可用的 conda 环境**（例如 **`mamba2`**）：

```powershell
conda activate mamba2
python -c "import torch; print(torch.__version__)"
cd D:\cursor_try\mamba2
```

若 **`conda`** 不在 **PATH**（如 **Cursor 内置终端**），可直接调用 env 解释器（本机示例）：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" -c "import torch; print(torch.__version__)"
```

若 **`conda activate mamba2`** 不可用：在 **Anaconda Prompt** 里 **`conda env list`** 找到 **带 torch 的 env**，再 **`conda activate <name>`**。

可选（Hub 限速）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## 2. 机制线 B-S2+（path reader 岭探针，CPU）

与 **path-batch** **分列**；见 **`RETRIEVAL_HEAD_NOTES.md`**、**`NEXT_EXPERIMENTS_COMMANDS.md` §6**。

**8 叶（较快）**：

```powershell
conda activate mamba2
cd D:\cursor_try\mamba2

python scripts/research/probe_path_reader_linear.py --cpu --n-leaves 8 --leaf-split heldout `
  --out-json results/metrics/probe_path_reader_linear_text8_heldout_local5060.json
```

**16 叶（与文档示例一致）**：

```powershell
python scripts/research/probe_path_reader_linear.py --cpu --n-leaves 16 --leaf-split heldout `
  --out-json results/metrics/probe_path_reader_linear_text16_heldout_local5060.json
```

可选 **BCE 训练 50 步**（与 ridge 对照）：

```powershell
python scripts/research/probe_path_reader_linear.py --cpu --n-leaves 16 --leaf-split heldout `
  --train-steps 50 --train-lr 3e-3 `
  --out-json results/metrics/probe_path_reader_linear_text16_heldout_train50_local5060.json
```

**已跑通例**（**`git_sha=498be11`**）：**`results/metrics/probe_path_reader_linear_text16_heldout_train50_local5060.json`**；登记 **X-20260410-local5060-bs2plus-train50-n16**。

跑通后：**`git add`** JSON，**`EXPERIMENT_REGISTRY.md`** **新开一行**（id 建议 **`X-local5060-bs2plus-…`**）。

---

## 2b. 任务线 A2-S3（叶对 cohort + ridge，CPU）

与 **path-batch**、**3090 `init_seed`×5** **分列**；见 **`docs/experiments/phases/PHASE1_MANUSCRIPT.md` §8.2**、**`docs/experiments/phases/PHASE2_DRAFT.md`**。

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts/research/task_wikitext_path_pair.py --cpu --num-leaves 8 --cohort sibling --chunk-len 8 --dim 128 `
  --out-json results/metrics/task_wikitext_sibling8_local5060_cpu_20260410.json
```

**已跑通例**（**`git_sha=1a65b29`**）：**`results/metrics/task_wikitext_sibling8_local5060_cpu_20260410.json`** — **stratified**，**`ridge_concat.*.test_acc`≈0.857**；登记 **X-20260410-local5060-a2s3-n8-strat**。

---

## 3. 系统线：Wikitext path-batch smoke（5060 CUDA 或 CPU）

**动机向**（**HF naive Mamba** 峰值可达 **GiB 级** — 与 **3090 fused** **禁止无脚注混表**）：

```powershell
conda activate mamba2
cd D:\cursor_try\mamba2
$STAMP = Get-Date -Format "yyyyMMddTHHmmZ"

python scripts/benchmarks/benchmark_wikitext_tree.py `
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 --warmup 2 --reps 5 `
  --out-json "results/metrics_result/benchmark_wikitext_local5060_smoke_${STAMP}_n8_c8.json"
```

去掉 **`--cpu`**（若脚本支持默认 CUDA）即走 **5060 GPU**；以 **`--cpu`** 可只做 **可跑性检查**（建议 **`--warmup 1 --reps 2`** 缩短墙钟；**`peak_alloc_mib`** 在 CPU 上 **常为 0**，**非** GPU 峰值动机）。

**已跑通例**（**`git_sha=a466f50`**）：**`results/metrics_result/benchmark_wikitext_local5060_cuda_20260410T1204Z_n8_c8.json`** — **Mamba2 `peak_alloc_mib`≈1201**（**HF naive**）；登记 **X-20260410-benchmark-wikitext-local5060-cuda-n8c8**。

**CPU 可跑性例**（**`git_sha=498be11`**）：**`results/metrics_result/benchmark_wikitext_local5060_cpu_20260410T1220Z_n8_c8.json`**（**`WARMUP=1` `REPS=2`**）；登记 **X-20260410-local5060-wikitext-cpu-n8c8**。

---

## 4. 辅线：SSGS × Wikitext（CPU，小模型）

```powershell
python scripts/research/demo_ssgs_mamba_wikitext.py --cpu --num-leaves 8 --chunk-len 4 --dim 64 --layers 2 `
  --out-json results/metrics/ssgs_mamba_wikitext_n8_smoke_local5060.json
```

**已跑通例**（**`git_sha=1a65b29`**，文件名带超参）：**`results/metrics/ssgs_mamba_wikitext_n8_c4_d64_local5060_20260410.json`** — **`ok True`**，**snapshots_taken 7**、**rollbacks 11**、**leaf_checks 8**；登记 **X-20260410-local5060-ssgs-wikitext-n8-c4d64**（与 **`metrics_result/ssgs_mamba_wikitext_*`** **归档 grid** **分列**）。

若 **torch** 仍异常，先修复 **conda env**，勿强用 **base**。

---

## 5. 回归（CI / 本地）

**无 torch（快）** — 聚合脚本单测：

```powershell
cd D:\cursor_try\mamba2
py -3 -m pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py -q
```

**全量（须 `mamba2` 或同等 torch 环境）**：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" -m pytest tests/ -q
```

**已验证**：**约 21 passed**（含 **`test_tf_kv_trajectory_l3`** 等须 **torch**；以 **`pytest tests/ -q`** 实际计数为准；耗时约 **15 s** 量级）。

### 5.1 可选：机制线 / §7 复跑（与 **`NEXT_RESEARCH_PLAN`** **「算力不可用时的备选推进」§B**）

**§7 S1（Mamba cache 段克隆）** — **CPU**：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts/research/benchmark_mamba2_cache_snapshot_segments.py --device cpu --depth 4 `
  --out-json results/metrics/mamba2_cache_snap_segments_depth4_cpu_local5060_confirm.json
```

**§7 S2（TF-R1 无 KV 重算）** — **CPU**（与 **S1** 同 depth/chunk/dim；**低 `warmup`/`reps`** 缩短墙钟）：

```powershell
python scripts/research/benchmark_tf_r1_path_segments.py --device cpu --depth 4 --warmup 1 --reps 5 `
  --out-json results/metrics/tf_r1_path_segments_depth4_cpu_local5060_confirm.json
```

**B-S2（GPT-2 层向量 + topic heldout）**：

```powershell
python scripts/research/probe_retrieval_correlation.py --cpu --model gpt2 --label-mode topic --topic-split heldout `
  --out-json results/metrics/probe_retrieval_linear_gpt2_topic_heldout_local5060_confirm.json
```

登记：**X-20260410-local5060-section7-s1-cache-confirm**、**X-20260410-local5060-section7-s2-tf-r1-cpu-confirm**、**X-20260410-local5060-b2-gpt2-topic-heldout-confirm**。

---

## 6. 与总清单的对应

**`docs/overview/execution/NEXT_RESEARCH_PLAN.md`**：**后续方向**、**当前收口清单**、**算力不可用时的备选推进**；**3090 登记级** 命令见 **`docs/environment/runbooks/NEXT_EXPERIMENTS_COMMANDS.md` §0–§10**（**服务器可用时**优先跑 **P1/P2** 与文档一致的新行 JSON）。
