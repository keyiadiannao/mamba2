# `scripts/engineering/`

| 脚本 | 说明 |
|------|------|
| **`run_engineering.py`** | **统一 CLI**（**§Ⅷ-1 G1**）：**`path-batch-smoke` \| `g3-compare` \| `causal-kv-smoke` \| `m1-ssgs-vs-kv`** → 转发既有 Runner |
| **`run_engineering_path_batch_smoke.py`** | **G1**：**`kind=engineering_path_batch_smoke`**，**`payload`** = **`benchmark_wikitext_tree`** |
| **`run_causal_lm_path_kv_smoke.py`** | **G3-a**（烟测）：**仅树路径文档** → **GPT-2** / **`--mamba`**（**`AntonV/mamba2-370m-hf`**）；**`kind=engineering_causal_lm_path_kv_smoke`** |
| **`run_g3_causal_lm_compare.py`** | **G3-b**：**baseline（全路径拼接单次 AR）** vs **树路径逐条 forward** → **`kind=engineering_causal_lm_compare`** → **`eng_g3_*.json`**（**主文独立表**，见 **`ENGINEERING_NORTH_STAR_PLAN.md` §4.3**） |

### G1（Sprint 1 · **`kind=engineering_path_batch_smoke`**）

**统一入口（推荐）**：

```bash
python scripts/engineering/run_engineering.py path-batch-smoke --out-json results/metrics_result/engineering/eng_path_batch_smoke.json
python scripts/engineering/run_engineering.py --help
```

需 **`datasets`** + 可 **`import torch`**（与 **`benchmark_wikitext_tree`** 相同）。**信封 schema** 单测无需 GPU：**`pytest tests/test_engineering_path_batch_smoke.py`**。

产出 JSON 请在 **AutoDL / 本机正常 CUDA 环境** 运行（Windows 若 **c10.dll** 等失败，请换 Linux）：

```bash
# 固定名（覆盖前请备份）
py -3 scripts/engineering/run_engineering_path_batch_smoke.py --out-json results/metrics_result/engineering/eng_path_batch_smoke.json
# 或带时间戳（推荐）
py -3 scripts/engineering/run_engineering_path_batch_smoke.py --out-json results/metrics_result/engineering/eng_path_batch_smoke_$(date -u +%Y%m%dT%H%MZ).json
```

**CPU 可跑通但较慢**：加 **`--cpu`**；默认 **n8 f2 c8 dim128**，与 G3/SSGS 并列烟测对齐。

跑通后把 **basename** 补进 **`EXPERIMENT_REGISTRY`** **`ENG-20260411-path-batch-smoke-v1`**。**已归档示例（AutoDL CUDA）**：**`results/metrics_result/engineering/eng_path_batch_smoke_cuda_20260411.json`**（**dim128**）。**dim256（Ⅵ-1 同 harness）**：**`--dim 256`** → 例 **`eng_path_batch_smoke_dim256_20260411T0850Z.json`**；**`ENG-20260411-path-batch-smoke-dim256-v1`**。

```bash
# G3-a — 需 transformers + HF 权重；默认仅 GPT-2
py -3 scripts/engineering/run_causal_lm_path_kv_smoke.py --out-json results/metrics_result/engineering/eng_causal_kv_gpt2.json
# 双臂（显存够时）
py -3 scripts/engineering/run_causal_lm_path_kv_smoke.py --mamba --out-json results/metrics_result/engineering/eng_causal_kv_both.json
```

**G3-b**（**独立实验、独立 `kind`**）：

（先 **`cd`** 到**本仓库根目录**，勿使用文档里的占位路径 **`/path/to/mamba2`**。）

```bash
export PYTHONPATH=.
STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/engineering/run_g3_causal_lm_compare.py --out-json "results/metrics_result/engineering/eng_g3_${STAMP}.json"
python scripts/engineering/run_g3_causal_lm_compare.py --mamba --out-json "results/metrics_result/engineering/eng_g3_both_${STAMP}.json"
```

**CLI 无 torch 自检**：**`pytest tests/test_run_g3_causal_lm_compare_cli.py -q`**（**`--help`** 在 **`import torch` 之前**解析）。

**预训主线补格**（**`--max-length`** / **`--chunk-len`**，**与 M1 c12 对齐** 等）：归档与 **`ENG-g3-pretrain-ablate-v1`** 见 **`EXPERIMENT_REGISTRY.md`**。

### M1（**`ssgs_vs_kv_tree_nav_wikitext`** · 产出落 **`engineering/`**）

与 **`SSGS_MAINLINE_M1.md`**、**`X-ssgs-vs-kv-tree-nav-m1`** **同协议**；**`--out-json`** 指向 **`results/metrics_result/engineering/`** 时登记 **`ENG-m1-ssgs-vs-kv-engineering-v1`**（与 **`G3` / path-batch 分列**）。

```bash
export PYTHONPATH=.
STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py --device cuda \
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 \
  --out-json "results/metrics_result/engineering/eng_m1_ssgs_vs_kv_n8_c8_cuda_${STAMP}.json"
```

**`chunk_len=12`**：加 **`--chunk-len 12`**，basename 建议含 **`n8_c12`**。

### 同建树 SSGS 并列（Sprint 2 · 与 G3 **分列** · **已完成**）

与 **`run_causal_lm_path_kv_smoke`** **同 Wikitext 建树默认**（**n8 f2 c8 dim128**）的 **`demo_ssgs_mamba_wikitext`** 归档示例：**`results/metrics_result/ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0717Z.json`**（**`ok`**、**snapshots_taken** **7**、**rollbacks** **11**、**`leaf_checks`** **8**）。登记：**`EXPERIMENT_REGISTRY`** **`ENG-20260411-ssgs-mamba-wikitext-aligned-v1`**；历史 **`X-*`** 行见 **`X-20260407-ssgs-mamba-wikitext-tree`**。

复跑：

```bash
py -3 scripts/research/demo_ssgs_mamba_wikitext.py --device cuda --out-json results/metrics_result/ssgs_mamba_wikitext_n8_c8_dim128_cuda_<STAMP>.json
```

（**CPU** 加 **`--cpu`**；**`<STAMP>`** 建议 UTC 时间戳。）

### CI（G4）

**GitHub Actions**：**`.github/workflows/engineering_tests.yml`**（**`pytest`** **`test_engineering_path_batch_smoke`** + **`test_causal_lm_kv_stats`** + **`test_run_g3_causal_lm_compare_cli`** + **`test_run_engineering_cli`**，**`PYTHONPATH`** = 仓库根）。推送 **`main`/`master`**、**PR**、**每日 06:00 UTC**、**手动 `workflow_dispatch`** 均会跑。

主计划：**`docs/overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md`**。
