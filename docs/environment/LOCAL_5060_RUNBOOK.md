# 本机 RTX 5060（Windows）推进手册

> **定位**：在 **服务器空闲前**，于 **本机** 跑 **轻量** 实验；产出 **JSON** 入 **`results/metrics/`**（或 **`metrics_result/`**），**登记** 时注明 **5060 / CPU / 本机** 与 **`git_sha`**。  
> **公平性**：**5060 + HF naive** 与 **3090 fused**、**path-batch** 与 **§7** / **SSGS** / **A2-S3** 须 **分列脚注**（**`FIGURE_CAPTIONS_STAGE1.md`**、**`PHASE1_MANUSCRIPT` §5.1**）。

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

跑通后：**`git add`** JSON，**`EXPERIMENT_REGISTRY.md`** **新开一行**（id 建议 **`X-local5060-bs2plus-…`**）。

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

去掉 **`--cpu`**（若脚本支持默认 CUDA）即走 **5060 GPU**；以 **`--cpu`** 可只做 **可跑性检查**。

**已跑通例**（**`git_sha=a466f50`**）：**`results/metrics_result/benchmark_wikitext_local5060_cuda_20260410T1204Z_n8_c8.json`** — **Mamba2 `peak_alloc_mib`≈1201**（**HF naive**）；登记 **X-20260410-benchmark-wikitext-local5060-cuda-n8c8**。

---

## 4. 辅线：SSGS × Wikitext（CPU，小模型）

```powershell
python scripts/research/demo_ssgs_mamba_wikitext.py --cpu --num-leaves 8 --chunk-len 4 --dim 64 --layers 2 `
  --out-json results/metrics/ssgs_mamba_wikitext_n8_smoke_local5060.json
```

若 **torch** 仍异常，先修复 **conda env**，勿强用 **base**。

---

## 5. 无 GPU 回归（CI / 本地）

```powershell
cd D:\cursor_try\mamba2
py -3 -m pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py -q
```

---

## 6. 与总清单的对应

**`NEXT_RESEARCH_PLAN.md`** 篇首 **「当前收口清单」**、**「本机 RTX 5060 可推进」** 表；**服务器** 空闲后再跑 **3090 登记级** 与 **`NEXT_EXPERIMENTS_COMMANDS.md` §0–§10** 云端命令。
