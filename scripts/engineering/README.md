# `scripts/engineering/`

| 脚本 | 说明 |
|------|------|
| **`run_engineering_path_batch_smoke.py`** | **G1**：**`kind=engineering_path_batch_smoke`**，**`payload`** = **`benchmark_wikitext_tree`** |
| **`run_causal_lm_path_kv_smoke.py`** | **G3 Sprint 2**：**同建树** 路径文档 → **GPT-2**（默认）与可选 **`--mamba`**（默认 **`AntonV/mamba2-370m-hf`**，非 **`state-spaces/mamba2-370m`**）；**`kind=engineering_causal_lm_path_kv_smoke`** |

### G1（Sprint 1 · **`kind=engineering_path_batch_smoke`**）

需 **`datasets`** + 可 **`import torch`**（与 **`benchmark_wikitext_tree`** 相同）。**信封 schema** 单测无需 GPU：**`pytest tests/test_engineering_path_batch_smoke.py`**。

产出 JSON 请在 **AutoDL / 本机正常 CUDA 环境** 运行（Windows 若 **c10.dll** 等失败，请换 Linux）：

```bash
# 固定名（覆盖前请备份）
py -3 scripts/engineering/run_engineering_path_batch_smoke.py --out-json results/metrics_result/engineering/eng_path_batch_smoke.json
# 或带时间戳（推荐）
py -3 scripts/engineering/run_engineering_path_batch_smoke.py --out-json results/metrics_result/engineering/eng_path_batch_smoke_$(date -u +%Y%m%dT%H%MZ).json
```

**CPU 可跑通但较慢**：加 **`--cpu`**；默认 **n8 f2 c8 dim128**，与 G3/SSGS 并列烟测对齐。

跑通后把 **basename** 补进 **`EXPERIMENT_REGISTRY`** **`ENG-20260411-path-batch-smoke-v1`**。**已归档示例（AutoDL CUDA）**：**`results/metrics_result/engineering/eng_path_batch_smoke_cuda_20260411.json`**。

```bash
# G3 — 需 transformers + HF 权重（首次下载大）；默认仅 GPT-2
py -3 scripts/engineering/run_causal_lm_path_kv_smoke.py --out-json results/metrics_result/engineering/eng_causal_kv_gpt2.json
# 双臂（显存够时）
py -3 scripts/engineering/run_causal_lm_path_kv_smoke.py --mamba --out-json results/metrics_result/engineering/eng_causal_kv_both.json
```

### 同建树 SSGS 并列（Sprint 2 · 与 G3 **分列** · **已完成**）

与 **`run_causal_lm_path_kv_smoke`** **同 Wikitext 建树默认**（**n8 f2 c8 dim128**）的 **`demo_ssgs_mamba_wikitext`** 归档示例：**`results/metrics_result/ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0717Z.json`**（**`ok`**、**snapshots_taken** **7**、**rollbacks** **11**、**`leaf_checks`** **8**）。登记：**`EXPERIMENT_REGISTRY`** **`ENG-20260411-ssgs-mamba-wikitext-aligned-v1`**；历史 **`X-*`** 行见 **`X-20260407-ssgs-mamba-wikitext-tree`**。

复跑：

```bash
py -3 scripts/research/demo_ssgs_mamba_wikitext.py --device cuda --out-json results/metrics_result/ssgs_mamba_wikitext_n8_c8_dim128_cuda_<STAMP>.json
```

（**CPU** 加 **`--cpu`**；**`<STAMP>`** 建议 UTC 时间戳。）

### CI（G4）

**GitHub Actions**：**`.github/workflows/engineering_tests.yml`**（**`pytest`** 上述两文件，**`PYTHONPATH`** = 仓库根）。推送 **`main`/`master`**、**PR**、**每日 06:00 UTC**、**手动 `workflow_dispatch`** 均会跑。

主计划：**`docs/overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md`**。
