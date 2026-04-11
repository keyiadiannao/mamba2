# 工程北星执行计划（正式开工版）

> **状态**：**2026-04-11** 起与 **战略 B**（**`PLAN_NOW_TO_DONE.md`** 篇首）对齐：**论文 1 主写作** 以 **§Ⅷ 门闩 G** 为界；**`docs/overview/engineering/`** 仅本文件为 **计划主文档**（**已删** 独立 **`README.md`**）；**命令速查** 见 **`scripts/engineering/README.md`**。  
> **门闩定义**：**`PLAN_NOW_TO_DONE.md` §Ⅷ**（**G1–G5**、**§Ⅷ-0 真 TF 定义**）。**§Ⅷ-1 G1 统一 CLI**：**`scripts/engineering/run_engineering.py`**。**文献动机与 RW 边界**（树状 RAG / 回溯文献）见 **`docs/research/RESEARCH_NOTES.md` §8**，避免再开新文件。

---

## 0. 索引（本目录）

| 章节 | 内容 |
|------|------|
| **§1** | 与旧线 **文档/脚本/结果/登记** 分列策略 |
| **§2** | 门闩 **G** 与交付物 |
| **§3** | **Sprint 1–3** |
| **§4** | **G5 公平对照一页纸**（**§4.2** 命令；**§4.3** **G3 独立**；**§4.4** **公平 Q1–Q5 提案**；不设独立 `FAIR_COMPARE_*.md`） |
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
| **G1** | **`run_engineering_path_batch_smoke.py`** + **`run_engineering.py`** **统一 CLI**（**`path-batch-smoke` / `g3-compare` / `causal-kv-smoke` / `m1-ssgs-vs-kv`**） | 同一 **`kind=engineering_path_batch_smoke`** 下封装 **`benchmark_wikitext_tree`**；**`run_engineering.py`** 仅 **转发**，**不**替代各 **`kind`**；**下一迭代** 拆 **HF `AutoModelForCausalLM` 臂** 入 **path-batch**（见 **`arms_note`**） |
| **G2** | 一条 **预训权重** 写入 **G1** 默认 config | **非** `build_toy_mamba2_for_ssgs` 随机初始化；**`from_pretrained` 或登记 checkpoint 名** |
| **G3** | **预训因果 LM 独立实验**（**§4.3**）：**标准自回归 vs 树路径文档** × **KV / SSM state / 峰值**；主文 **独立表**，**非** path-batch 脚注 | **`kind=engineering_causal_lm_compare`**（**G3-b**）；**G3-a** 烟测见下 **Sprint 2**；与 **SSGS / 玩具 path-batch** **分表** |
| **G4** | **`pytest`** 或 **nightly** 子集 + **README 一键** | 新环境 **15 分钟内** 跑出 **smoke JSON**（CPU 可降维） |

---

## 3. Sprint 划分（建议节奏）

### Sprint 1（第 1–2 周）：契约 + G1 骨架

- [x] **G5**：**§4** 已定 **gpt2 / `AntonV/mamba2-370m-hf`**（**§4** 表）。  
- [x] **目录与占位**：**`results/metrics_result/engineering/`**；**`src/rag_tree/engineering_envelope.py`**（无 torch）。  
- [x] **G1 最小实现**：**`run_engineering_path_batch_smoke.py`** → **`kind=engineering_path_batch_smoke`**，**`payload`** = **`benchmark_wikitext_tree`** 全表；单测 **`tests/test_engineering_path_batch_smoke.py`**。  
- [x] **登记与命令**：**`EXPERIMENT_REGISTRY.md`** **`ENG-20260411-path-batch-smoke-v1`** + **`scripts/engineering/README.md`** **§G1**（**AutoDL 可复制一行**）。  
- [x] **`eng_path_batch_smoke_*.json` 入仓**：**`results/metrics_result/engineering/eng_path_batch_smoke_cuda_20260411.json`**（**`ENG-20260411-path-batch-smoke-v1`**）。  
- [x] **dim256 同 harness 归档**（**Ⅵ-1**）：**`eng_path_batch_smoke_dim256_20260411T0850Z.json`**；**`ENG-20260411-path-batch-smoke-dim256-v1`**（与 **dim128**、**G3** **分列**）。  
- [x] **工程单测（无 GPU）**：**`pytest tests/test_engineering_path_batch_smoke.py`** **+** **`tests/test_causal_lm_kv_stats.py`** **+** **`tests/test_run_engineering_cli.py`**（与 Sprint 3 **G4** 预热一致）。

### Sprint 2（第 3–5 周）：预训权重 + G3（**G3-a 烟测 / G3-b 独立表**）

- [x] **Hub id**：见 **§4**（**gpt2** / **`AntonV/mamba2-370m-hf`**；**G2「预训接入」** 对 Mamba 臂已由该 Hub **`from_pretrained`** 满足，**非** 玩具随机初始化）。  
- [x] **G3-a 烟测**（**`kind=engineering_causal_lm_path_kv_smoke`**）：**`run_causal_lm_path_kv_smoke.py`** — **仅树路径文档** **`iter_path_documents`** → **`AutoModelForCausalLM`** **`use_cache=True`** → **`peak_alloc_mib`** + **`past_or_cache_nbytes`**；归档 **`eng_causal_kv_both.json`**；**`ENG-20260411-causal-lm-path-kv-smoke-v1`**。  
- [x] **G3-b Runner**（**§4.3**）：**`run_g3_causal_lm_compare.py`** — **① baseline**（全路径文档拼接、单次截断前向）+ **② 树路径逐条前向**；**`kind=engineering_causal_lm_compare`**；CLI 单测 **`tests/test_run_g3_causal_lm_compare_cli.py`**。  
- [x] **G3-b 归档**：**`eng_g3_20260411T0815Z.json`**（**gpt2 · n8**）、**`eng_g3_both_20260411T0821Z.json`**（**n8 双臂**）、**`eng_g3_both_n16_20260411T0837Z.json`**、**`eng_g3_both_n32_20260411T0837Z.json`**（**叶扫**）；登记 **`ENG-g3-causal-lm-compare-v1`**。**预训补格**（**`ENG-g3-pretrain-ablate-v1`**）：**ml256 / ml1024 / c12** 见 **`eng_g3_both_n8_ml256_*`**、**`…_ml1024_*`**、**`…_n8_c12_*`**；**G3-a c12**：**`eng_causal_kv_both_c12_*`**。**叙事**：**ml512·c8/c12** 下 Mamba **≈1.11**；**ml256** **≈1.20**；**ml1024** **ratio=1.0**（**非**全序列长度普适常数）—— 见 **§4.3**。  
- [x] 与 **`demo_ssgs_mamba_wikitext`** **同建树** 的 **并列 smoke**：**`ssgs_mamba_wikitext_n8_c8_dim128_cuda_20260411T0717Z.json`**；**`ENG-20260411-ssgs-mamba-wikitext-aligned-v1`**。

### Sprint 3（第 6–8 周）：G4 硬化 + 可选接 SSGS

- [x] **CI/nightly**：**`.github/workflows/engineering_tests.yml`** — **`pytest`** **`tests/test_engineering_path_batch_smoke.py`** + **`tests/test_causal_lm_kv_stats.py`** + **`tests/test_run_g3_causal_lm_compare_cli.py`** + **`tests/test_run_engineering_cli.py`**（**无 torch**；**push/PR** + **每日 06:00 UTC** **`workflow_dispatch`**）。  
- [x] **M1 落工程目录**：**`results/metrics_result/engineering/eng_m1_ssgs_vs_kv_n8_c8_cuda_20260411T0907Z.json`**、**`…_n8_c12_cuda_20260411T0907Z.json`**（**`kind=ssgs_vs_kv_tree_nav_wikitext`**）；登记 **`ENG-m1-ssgs-vs-kv-engineering-v1`**（与 **`X-ssgs-vs-kv-tree-nav-m1`** **分列 basename**，协议同）。  
- [x] **统一 CLI**：**`scripts/engineering/run_engineering.py`**（**§Ⅷ-1 G1**）— **子命令** **`path-batch-smoke`**、**`g3-compare`**、**`causal-kv-smoke`**、**`m1-ssgs-vs-kv`** → 既有脚本；**`pytest`** **`tests/test_run_engineering_cli.py`**。**`kind`** 仍分属各 Runner，**不**混 JSON。

**门闩 G 已达成**（**2026-04-11**）：**G1–G5** 证据与勾选见上；**`PLAN_NOW_TO_DONE.md`** 篇首与 **§Ⅷ** 已声明 **战略 B** 下 **门闩后再启 §Ⅰ**。**启用 §Ⅰ 主稿前** 请 **确认 GitHub Actions `engineering-tests` 绿**（与推送 **HEAD** 一致）。

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

### 4.2 可复现命令（与 **`PLAN_NOW_TO_DONE.md` §Ⅷ-0「可复现」** 对齐）

**`git_sha`** 以各 **`results/metrics_result/**/*.json`** 内字段为准；环境见 **`environment/requirements.txt`**、**`environment/MAMBA2.md`**。详注与变体：**`scripts/engineering/README.md`**。

| 用途 | 命令（仓库根目录） |
|------|---------------------|
| **G1** path-batch 信封（**`kind=engineering_path_batch_smoke`**） | `python scripts/engineering/run_engineering_path_batch_smoke.py --out-json results/metrics_result/engineering/eng_path_batch_smoke_<STAMP>.json`（**GPU**：不传 **`--cpu`**；需 **`datasets`** + **`torch`**） |
| **G3** 真 TF 栈：因果 LM **KV / 峰值**（**`kind=engineering_causal_lm_path_kv_smoke`**） | `python scripts/engineering/run_causal_lm_path_kv_smoke.py --mamba --out-json results/metrics_result/engineering/eng_causal_kv_both.json`（**仅 GPT-2**：去掉 **`--mamba`**；需 **`transformers`**） |
| **G4** 工程单测（**无 GPU / 无 Hub 下载**） | `PYTHONPATH=. python -m pytest tests/test_engineering_path_batch_smoke.py tests/test_causal_lm_kv_stats.py -q`（与 **`.github/workflows/engineering_tests.yml`** 一致） |

**说明**：**`PLAN_NOW_TO_DONE.md` §Ⅷ-1** 表内 **G1「统一 Runner」** 仍可能要求 **单 CLI 多臂**；当前仓库以 **分列脚本 + 同一建树模块** 落地，**不在此节宣称** 已完成 **§Ⅷ-1** 全文义；**§4 主表 + 本节** 满足 **§Ⅷ-0** 可写进论文/附录脚注，**§Ⅷ-1 G5「公平性文档」子门闩** 可标 **已冻结**（见 **`PLAN_NOW_TO_DONE.md` §Ⅷ** 旁注）。

### 4.3 G3 **独立实验**（预训因果 LM）：研究问题 · 协议 · 主表（**非**「附录级工程脚注」）

**定位**：**不是**「为写文章而做研究」。审稿在 **玩具 trunk（128d / 随机初始化）** 之外必然会问 **预训权重下** 的行为；本实验直接回答：**预训因果 LM 在树路径文档批处理 vs 标准自回归下，KV / SSM 状态与峰值显存相对各自 baseline 的偏离** —— **值得单独一张表 + Discussion**，**reviewer 不能当脚注忽略**。

**与 path-batch 主表关系**：**禁止**与 **`benchmark_wikitext_tree`** 玩具表 **合并数值**（harness 不同：**随机 trunk** vs **全量预训解码栈**）。**跨表** 仅 **分列 + Discussion**，**不做无脚注的跨表直接数值胜负**。

**研究问题（锚定）**：

> **预训因果 LM（GPT-2 124M / Mamba2 370M 级）在「树路径拼接文档」与「标准自回归单序列」两种输入模式下，缓存或状态占用与峰值显存相对各自标准解码 baseline 的偏离，是否呈现可报告的系统差异？**

**模型 Hub（默认可加载）**：

| 臂 | Hub id | 备注 |
|----|--------|------|
| **GPT-2** | **`openai-community/gpt2`** | 124M **decoder-only** |
| **Mamba2** | **`AntonV/mamba2-370m-hf`** | 与 **state-spaces** 预训对齐的 **HF 镜像**（含 **`model_type`**、tokenizer）；**`state-spaces/mamba2-370m`** 无标准 HF **`config`/分词器**，**不作** 本实验默认 |

**协议（须写死进 JSON + 正文方法节）**：

| 维度 | 约定 |
|------|------|
| **输入模式** | **① 标准自回归**：单序列 **`input_ids`**，长度至 **`max_length`**（baseline）；**② 树路径批**：与 **①** 同 **`max_length`**、同 tokenizer，文档来自 **`iter_path_documents`**（**同建仓树**：**`num_leaves` / `fanout` / `chunk_len`** 网格可扫） |
| **度量** | **`torch.cuda.max_memory_allocated` 峰值**；GPT-2：**`past_or_cache_nbytes`** 或 **§4** 粗算估计；Mamba2：**cache/state 字节**（能测则测，**0** 时脚注 HF 行为）；**墙钟** 仅 **同机、同臂、同输入模式** 内对比 |
| **控制** | 同 **GPU**、同 **`dtype`**、同 **`git_sha`**；**batch** 语义与实现写进 JSON |
| **主表关键列** | 每模型内 **「树路径 / baseline」偏离**（比值或差），而非仅绝对 MiB；**124M vs ~370M** **不对等** 在 **Discussion** 用 **MiB/M params** 等 **归一化讨论**（**默认不** 为等参截断预训层；若审稿强要求再补 **`gpt2-medium`** 等） |

**交付物**：

| 层级 | **`kind`** | 脚本 | 结果示例 |
|------|------------|------|----------|
| **G3-a**（已有） | **`engineering_causal_lm_path_kv_smoke`** | **`run_causal_lm_path_kv_smoke.py`** | **`eng_causal_kv_both.json`** |
| **G3-b** | **`engineering_causal_lm_compare`** | **`run_g3_causal_lm_compare.py`** | **`results/metrics_result/engineering/eng_g3_<STAMP>.json`** |

**登记**：**`ENG-*`** 一行 **`g3-causal-lm-compare-v1`**（basename 待 **G3-b** 跑通后填）。

### 4.4 公平维度 **Q1–Q5**（**导师会提案** · **已定稿入仓**，**待会议确认**）

以下为 **带去导师会的明确提案**（与 **`RESEARCH_STATUS_AND_DIRECTION.md` §6**、**§4**、**§4.3** 一致）；**非**代导师裁决，**会后若改一条** 应修订本节并 **`PLAN_NOW_TO_DONE.md` §Ⅵ** 互链。

---

**Q1 — 主表锁哪两项预算？墙钟放哪？**

- **提案**：**主表 = 峰值显存 + token 上界**；其中 **token** 建议 **两条都报**：**单路径 token 数** 与 **整次 forward 相关总 token 步**（或等价汇总），避免「峰值大是因为序列更长」类质疑。  
- **墙钟**：**不作为主表决策列**（避免叙事被「谁更快」带偏；与同机 path-batch 归档一致：**fused Mamba 墙钟未必优于 TF/GRU**）。**仍须报告**：放在 **脚注 / 补充列**，并标注 **仅同臂、同机、同 harness 可比**；**Discussion** 可写一句 **诚实结论**（例如 fused 与 TF/GRU 持平、**不支持**「Mamba 更快」作为主声称）。  
- **理由摘要**：峰值是**硬锚**；token 上界是解释峰值差异的**必要协变量**；墙钟在同臂内对 **naive vs fused** 仍有工程价值。

---

**Q2 — 主验证轴 vs G3 地位**

- **提案**：**主验证轴** = **path-batch 玩具 trunk + SSGS/M1 机制**（闭环 harness）**不变**。  
- **G3** = **独立实验表 + 独立小节**（**§4.3**），**不**与主表合并数字、**不**共用 **「Table X」**；**分列讨论**。  
- **理由**：预训 LM 与 toy trunk 在参数量、维度、层数、接口上**全不同**；同表并列会引发**无信息量**的数值大小比较。

---

**Q3 — 跨 harness 混表**

- **提案**：**禁止**：**脚本 / 模型来源 / HF 接口** 任一不同 → **必须分表**；脚注可互引（「见 Table Y」），**禁止**同表并列可比数字。  
- **灰色地带（将来）**：若 **同一预训模型、同一 `AutoModelForCausalLM` 接口** 下同时跑 **标准 forward** 与 **SSGS 控制流**，可进**同一张 G3 系表**；**仍禁止** toy trunk SSGS 与预训 GPT-2 forward 混表。

---

**Q4 — 下一阶段扫参**

- **提案**：**扫参严格在同一 harness 内**（例：path-batch 内扫 **dim / depth / leaves**；G3 内扫 **树规模**）。  
- **跨 harness**：**禁止跨 harness 数值比较表**；**Discussion** 仅允许**定性**趋势（各自 harness 内随规模的变化），**不**做跨架构排名。

---

**Q5 — 审稿人追问「公平」时的标准回应**

- **中文（导师会 / 方法脚注）**：  
  **「本文在同一实验协议内比较代价结构，不跨架构排名。不同实验线（玩具 path-batch、SSGS/M1 回溯、预训练因果 LM）使用不同的 harness 与模型配置，故分表报告。参数量不对等（如 GPT-2 124M vs Mamba-2 370M）已在 §4.3 / Discussion 显式声明；主结论不依赖跨架构绝对数值比较，而依赖同一协议下的行为差异。」**

- **英文（Discussion / rebuttal）**：  
  **"Our comparisons are intra-protocol: within each experimental harness, we control for tree structure, batch configuration, and measurement procedure. Cross-harness numerical comparisons are deliberately avoided in our tables. The parameter-count asymmetry (e.g., GPT-2 124M vs Mamba-2 370M) is stated in §4.3; our conclusions rest on behavioral differences within each protocol, not on cross-architecture absolute-value rankings."**

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
| 2026-04-11 | 已推送 **origin/master** **`a01d899`**；请在 **GitHub → Actions** 确认 **engineering-tests** 绿 |
| 2026-04-11 | **§4.2** 可复现命令；**G5**（**§Ⅷ-0 / §Ⅷ-1 公平性文档**）**冻结** — 旁注 **`PLAN_NOW_TO_DONE.md` §Ⅷ** |
| 2026-04-11 | **§4.3**：**G3** 升为 **独立实验**（**G3-a** 烟测 / **G3-b** **`run_g3_causal_lm_compare.py`**）；与 **path-batch 玩具** **分表**；**124M vs 370M** 讨论口径 |
| 2026-04-11 | **§4.4**：公平 **Q1–Q5** **导师会提案**（主表峰值+token；墙钟脚注；分表；同 harness 扫参；**中/英** 标准回应） |
| 2026-04-11 | **G3-b**：**`run_g3_causal_lm_compare.py`** 落地（**`kind=engineering_causal_lm_compare`**）；**`pytest`** **`test_run_g3_causal_lm_compare_cli`** |
| 2026-04-11 | **G3-b 归档**：**`eng_g3_20260411T0815Z.json`**、**`eng_g3_both_20260411T0821Z.json`**（**AutoDL** **`6fa7873`**）；**`ENG-g3-causal-lm-compare-v1`** |
| 2026-04-11 | **G3-b 叶扫**：**`eng_g3_both_n16_20260411T0837Z.json`**、**`eng_g3_both_n32_20260411T0837Z.json`**（**Mamba** **`peak_max_tree_over_baseline`** **≈1.111** **n8∥n16∥n32**） |
| 2026-04-11 | **G1 · dim256 path-batch**：**`eng_path_batch_smoke_dim256_20260411T0850Z.json`**；**`ENG-20260411-path-batch-smoke-dim256-v1`** |
| 2026-04-11 | **G3 预训补格**：**`ENG-g3-pretrain-ablate-v1`**（**ml256 / ml1024 / c12** + **G3-a c12**）；**Mamba ratio**：**ml512≈1.11**、**ml256≈1.20**、**ml1024=1.0** |
| 2026-04-11 | **M1 → `engineering/`**：**`eng_m1_ssgs_vs_kv_n8_c8_cuda_20260411T0907Z.json`**、**`…_n8_c12_cuda_20260411T0907Z.json`**；**`ENG-m1-ssgs-vs-kv-engineering-v1`** |
| 2026-04-11 | **§Ⅷ-1 G1**：**`run_engineering.py`** 统一 CLI（**四子命令** 转发）；**`test_run_engineering_cli`** |
| 2026-04-11 | **门闩 G 已达成** 段；**启用 §Ⅰ** 前 **CI `engineering-tests` 绿** |
