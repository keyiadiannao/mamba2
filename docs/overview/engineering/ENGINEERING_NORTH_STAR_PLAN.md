# 工程北星执行计划（正式开工版）

> **状态**：**2026-04-11** 起与 **战略 B**（**`PLAN_NOW_TO_DONE.md`** 篇首）对齐：**论文 1 主写作** 以 **§Ⅷ 门闩 G** 为界；**`docs/overview/engineering/`** 仅本文件为 **计划主文档**（**已删** 独立 **`README.md`**）；**命令速查** 见 **`scripts/engineering/README.md`**。  
> **门闩定义**：**`PLAN_NOW_TO_DONE.md` §Ⅷ**（**G1–G5**、**§Ⅷ-0 真 TF 定义**）。**文献动机与 RW 边界**（树状 RAG / 回溯文献）见 **`docs/research/RESEARCH_NOTES.md` §8**，避免再开新文件。

---

## 0. 索引（本目录）

| 章节 | 内容 |
|------|------|
| **§1** | 与旧线 **文档/脚本/结果/登记** 分列策略 |
| **§2** | 门闩 **G** 与交付物 |
| **§3** | **Sprint 1–3** |
| **§4** | **G5 公平对照一页纸**（**填写区**；不设独立 `FAIR_COMPARE_*.md`） |
| **§5** | **修订记录** |

---

## 1. 与「之前工作」的关系：分开什么、复用什么

### 1.1 为什么要分开

| 维度 | **既有线（阶段 0–5）** | **工程北星线（本计划）** |
|------|-------------------------|---------------------------|
| **目标** | 可发表 **机制证据链**、**七轴分列**、登记 **`X-*`** | **统一 Runner**、**HF 预训栈**、**真 TF 臂可比**、**CI smoke** |
| **文档** | **`docs/experiments/`**、**`NARRATIVE`**、**`PHASE1_*`** | **`docs/overview/engineering/`**（本目录）；**不**改写旧稿结论，仅 **互链** |
| **脚本** | **`scripts/research/`**、**`scripts/benchmarks/`**、**`scripts/server/`** | **`scripts/engineering/`** 新增 **入口**；**实现上 = 调旧模块** |
| **结果** | **`results/metrics_result/*.json`**（**`kind=`** 已有多类） | 默认 **`results/metrics_result/engineering/`** 或 basename **`eng_*` / `*_engineering_*`**；JSON 内 **`kind=engineering_path_arm_compare`**（名以首版实现为准） |
| **登记** | **`EXPERIMENT_REGISTRY.md`** **`X-*` / `A-*`** | 新增行前缀 **`ENG-`**（例 **`ENG-202604-runner-v1`**），与 **`X-*`** **分列**，避免混行 |

**调用关系（必须遵守）**：新 Runner **只 import** 现有 **`build_bottom_up_text_tree`**、**`benchmark_wikitext_tree`**、**`demo_ssgs_mamba_wikitext`**、**`benchmark_ssgs_vs_kv_tree_nav_wikitext`** 等；**禁止** 大段复制粘贴进 `scripts/engineering/`（避免双源真理）。

### 1.2 仍共享的「单一权威」

- **公平与层级**：**`RESEARCH_STATUS_AND_DIRECTION.md` §6**、**`PLAN_NOW_TO_DONE` §Ⅵ-0**。  
- **环境**：**`AUTODL_SETUP.md`**、**`scripts/server/_autodl_env.sh`**。  
- **旧 JSON**：工程线跑出的对比仍可 **脚注引用** **`X-ssgs-vs-kv-tree-nav-m1`** 等作为 **基线**，**新表另列** **`ENG-*`**。

---

## 2. 门闩 G 与交付物（与 §Ⅷ 一致，可勾选）

| 门闩 | 交付物 | 完成判据（客观） |
|------|--------|------------------|
| **G5** | **§4 一页纸**（本文件内） | **§Ⅷ-0** 四行表填满：**TF checkpoint**、**任务**、**预算锁两项**、**可复现命令** |
| **G1** | **`scripts/engineering/run_engineering_path_batch_smoke.py`**（**Sprint 1** 已落地 **信封 + path-batch 单格**） | 同一 **`kind=engineering_path_batch_smoke`** 下封装 **`benchmark_wikitext_tree`**；**下一迭代** 拆 **HF `AutoModelForCausalLM` 臂**（见 **`arms_note`**） |
| **G2** | 一条 **预训权重** 写入 **G1** 默认 config | **非** `build_toy_mamba2_for_ssgs` 随机初始化；**`from_pretrained` 或登记 checkpoint 名** |
| **G3** | **真 TF 臂** 的 **KV/重算/峰值** 或协议规定的等价量 | 与 **SSGS 迹** **分列** 同一 JSON 或 **双文件 + 登记一行** |
| **G4** | **`pytest`** 或 **nightly** 子集 + **README 一键** | 新环境 **15 分钟内** 跑出 **smoke JSON**（CPU 可降维） |

---

## 3. Sprint 划分（建议节奏）

### Sprint 1（第 1–2 周）：契约 + G1 骨架

- [x] **G5**：**§4** 已定 **gpt2 / `AntonV/mamba2-370m-hf`**（**§4** 表）。  
- [x] **目录与占位**：**`results/metrics_result/engineering/`**；**`src/rag_tree/engineering_envelope.py`**（无 torch）。  
- [x] **G1 最小实现**：**`run_engineering_path_batch_smoke.py`** → **`kind=engineering_path_batch_smoke`**，**`payload`** = **`benchmark_wikitext_tree`** 全表；单测 **`tests/test_engineering_path_batch_smoke.py`**。  
- [x] **登记与命令**：**`EXPERIMENT_REGISTRY.md`** **`ENG-20260411-path-batch-smoke-v1`** + **`scripts/engineering/README.md`** **§G1**（**AutoDL 可复制一行**）。  
- [x] **`eng_path_batch_smoke_*.json` 入仓**：**`results/metrics_result/engineering/eng_path_batch_smoke_cuda_20260411.json`**（**`ENG-20260411-path-batch-smoke-v1`**）。  
- [x] **工程单测（无 GPU）**：**`pytest tests/test_engineering_path_batch_smoke.py`** **+** **`tests/test_causal_lm_kv_stats.py`**（与 Sprint 3 **G4** 预热一致）。

### Sprint 2（第 3–5 周）：G2 预训 + G3 真 TF 读数

- [x] **Hub id**：见 **§4**（**gpt2** / **HF 兼容 Mamba2 检查点**）。  
- [x] **G3 烟测脚本**：**`run_causal_lm_path_kv_smoke.py`** — **同建树** **`iter_path_documents`** → **`AutoModelForCausalLM`** **`use_cache=True`** → **`peak_alloc_mib`** + **`past_or_cache_nbytes`**；**`src/rag_tree/causal_lm_kv_stats.py`**；单测 **`tests/test_causal_lm_kv_stats.py`**（**无 torch**）。  
- [x] **跑通归档**：**`results/metrics_result/engineering/eng_causal_kv_both.json`**（**`--mamba`** 双臂）；**`EXPERIMENT_REGISTRY`** 行 **`ENG-20260411-causal-lm-path-kv-smoke-v1`** 已补 **basename** 与结论（**JSON 可由 AutoDL 生成后 `scp` 入仓或仅登记路径**）。  
- [x] 与 **`demo_ssgs_mamba_wikitext`** **同建树** 的 **并列 smoke**（**脚注分列**）：**`results/metrics_result/ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0717Z.json`**；**`EXPERIMENT_REGISTRY`** **`ENG-20260411-ssgs-mamba-wikitext-aligned-v1`**。

### Sprint 3（第 6–8 周）：G4 硬化 + 可选接 SSGS

- [x] **CI/nightly**：**`.github/workflows/engineering_tests.yml`** — **`pytest`** **`tests/test_engineering_path_batch_smoke.py`** + **`tests/test_causal_lm_kv_stats.py`**（**无 torch**；**push/PR** + **每日 06:00 UTC** **`workflow_dispatch`**）。  
- [ ] （可选）把 **SSGS 迹** 并入 **同一 Runner** 的 **子命令**，**`kind`** 扩展或 **双 JSON**。

**门闩 G 达成**：**G1–G5 全勾** → 在 **`PLAN_NOW_TO_DONE.md` 修订记录** 写一行，并宣布 **启用 §Ⅰ 作为论文 1 成稿流水线**。

---

## 4. G5：公平对照一页纸（**填写区 · 与 §Ⅷ-0 一致**）

| 项 | **已定（推荐）** | **备选 / 备注** |
|----|------------------|-----------------|
| **TF / HF 臂（G3：KV 可量化）** | **`openai-community/gpt2`**（124M，12L，768d，**decoder-only**）：`AutoModelForCausalLM.from_pretrained("openai-community/gpt2")`；**`past_key_values`** 与 **`forward`** 语义与 **§Ⅷ-0**「真 TF」一致 | **`google/flan-t5-small`**：**encoder–decoder**，与 **同路径因果前向** 叙事 **易错位**，**不推荐** 作 Sprint 2 主对照 |
| **Mamba 臂（Sprint 2 预训）** | **`AntonV/mamba2-370m-hf`**（370M 级，`model_type: mamba2`，1024d，safetensors + tokenizer）：HF **`Mamba2ForCausalLM`**（**`transformers`** 需支持 Mamba2）。旧版 **`state-spaces/mamba2-370m`** 无 **`model_type`**、无分词器，**不**作 Sprint 2 默认 id | **`mamba2-2.7b`**：**Phase M2 / scale-up**，**不阻塞** **G5→G1→G2** 烟测链 |
| **Sprint 1（不变）** | **玩具 trunk**：**`Mamba2PathReader`（`Mamba2Model` + `inputs_embeds`）** + 同 **`benchmark_wikitext_tree`** 网格 | 与 **预训臂** **分列登记** |
| **任务** | **Sprint 1**：Wikitext-2 浅树 **path-batch**（**`benchmark_wikitext_tree`** 同参）；**SSGS/M1** **脚注分列** | **Sprint 2+**（可选）：**同建树** 上 **zero-shot / few-shot 叶对分类** 等 **任务指标** — **新 `kind`** |
| **预算（锁两项）** | **①** 路径上 **token 步数上界**：**`L_path ≤ H × T_node`**（**H** = 树高/边数，**`T_node`** = 每节点 chunk 展开词元数，由 **`build_bottom_up_text_tree`** 与 **chunk 设置** 决定）；**②** 单次基准内 **`torch.cuda.max_memory_allocated` 增量峰值**（**3090** 物理上限 **24 GiB** 作 **环境脚注**） | — |
| **不对等声明** | **370M Mamba-2 vs 124M GPT-2** 约 **3:1 参数量**，**非等参** —— **分列 `ENG-*`**，**Discussion** 一句显式说明；**G5 目标**是 **可复现工程基线 + 协议对齐**，**非** 宣称 **iso-param SOTA** | **若强追求等参**：裁层 **Mamba-2** 至与 **12L** 对齐会 **破坏** 预训权重完整性 → **新 confounder**；**默认不做**，除非单独 **消融脚注** |

### 4.1 批判性备注（仓库已核对）

1. **「TF 臂 = GPT-2」与当前 path-batch 主表**：阶段 1 **`benchmark_wikitext_tree`** 里 **Transformer 路径** 是 **`TransformerPathReader`（小型 Encoder 式 trunk）**，**不是** **`gpt2` 全解码栈**。因此：**Sprint 1 信封 JSON 不能**被说成「已与 GPT-2 对比」；**GPT-2 写入 G5** 是为 **Sprint 2 新子 harness**（**`forward` + `past_key_values`** 统计 **KV/峰值**）预留 **唯一 ID**，**实现落在 `scripts/engineering/` 新入口**，**调用** 仍与 **同建树** 对齐。  
2. **「Mamba 臂 = `AntonV/mamba2-370m-hf`」与当前 `Mamba2PathReader`**：玩具 trunk **dim=128**；**370M** 为 **1024d** —— Sprint 2 必须约定 **投影层 / 换 embedding 维 / 换 chunk 编码管道** 之一，并在 **`ENG-*`** **一行**写清，**禁止**与 **Sprint 1 `payload`** 无脚注混表。  
3. **选型合理性**：**GPT-2 small** 作 **HF 因果基线**、**`AntonV/mamba2-370m-hf`**（与 **state-spaces** 预训对齐的 HF 镜像）作 **Mamba2 因果** —— 在 **审稿可读性** 与 **3090 显存** 上 **可接受**；**3:1 不对等** 用 **分列 + Discussion** 处理 **优于** 盲目裁层。

---

## 5. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：**文档/脚本/结果/登记** 与旧线分离策略；**Sprint 1–3**；**G5→G1** 顺序；互链 **§Ⅷ** |
| 2026-04-11 | **§0** 索引；**§4** 合并 **G5**（**不设**独立 `FAIR_COMPARE_*.md`）；**G1** 落地 **`run_engineering_path_batch_smoke.py`**；互链 **`RESEARCH_NOTES` §8** |
| 2026-04-11 | **§4**：**G5 定稿** — **gpt2** / **mamba2-370m**；**§4.1** 批判性备注（path-batch 与 **GPT-2 全栈** 区分、**1024d** 投影义务） |
| 2026-04-11 | **Sprint 2**：**`run_causal_lm_path_kv_smoke.py`** + **`causal_lm_kv_stats.py`** |
| 2026-04-11 | **§4 G5**：Mamba 默认 **`AntonV/mamba2-370m-hf`**；**G3** 归档 **`eng_causal_kv_both.json`**；**`ENG-20260411-ssgs-mamba-wikitext-aligned-v1`**：**`ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0717Z.json`**（与 **G3** 同树参 **分列**） |
| 2026-04-11 | **Sprint 1**：**`README.md` §G1**；**`pytest`** **`test_engineering_*`** + **`test_causal_lm_kv_stats`**（**8 passed**，无 GPU）；**`eng_path_batch_smoke_cuda_20260411.json`** 入仓；**`ENG-20260411-path-batch-smoke-v1`** 结案 |
| 2026-04-11 | **Sprint 3 / G4**：**`engineering_tests`** GitHub Actions（**无 torch** 工程单测） |
