# 从现在到结题：多步总览（滚动）

> **用途**：给你一张 **方向卡** — 每步 **做什么、验证什么、产出什么**；细节命令仍以 **`NEXT_EXPERIMENTS_COMMANDS.md`**、**`SSGS_MAINLINE_M1.md` §6**、**`SUBMISSION_PACK.md`** 为准。  
> **当前逻辑阶段**：**阶段 5（成文）**（**`RESEARCH_PHASES_0_TO_DONE.md`**）；实证 **0–4** 已收口，**主线不再依赖新实验也能投稿**。  
> **修订**：截稿或每完成一大步，在文末 **修订记录** 写一行；**`CURRENT_SPRINT.md`** 篇首可互链本文件。

---

## 总流程（五段）

```mermaid
flowchart LR
  A[阶段5 成文] --> B[数据冻结]
  B --> C[可选 M2 实验]
  C --> D[截稿 / 投出]
  D --> E[审稿 / 远期]
```

| 段 | 名称 | 你要验证什么 | 完成标志（客观） |
|----|------|----------------|------------------|
| **Ⅰ** | **成文 P0** | 叙事 **≤ L1–L3**（**`RESEARCH_STATUS` §3.5**）；**七轴不混读** | 正稿含 **§A3 类脚注**；**§A2 basename** 与登记册 **人工逐字** 一致 |
| **Ⅱ** | **数据与仓库** | 可复现路径、**`json_path`** 为仓内 POSIX、单测绿 | **`git status`** 干净；**`aggregate_*` stdout** 的 **N** 与 CSV 一致；**AutoDL** **`python -m pytest tests/ -q`** 通过 |
| **Ⅲ** | **可选 M2**（与 Ⅰ 并行） | **M1** 在 **c12** 或 **同树 bundle** 上仍 **可实现**；**非**新对比法默认 | 新 **JSON + STAMP**；**`EXPERIMENT_REGISTRY`** / **`DATA_ARCHIVE`** 补一行或一句 |
| **Ⅳ** | **截稿** | 无未引用路径、无跨轴混表 | 按 venue 提交；本地留 **tag 或 commit** 对齐 **`git_sha`** |
| **Ⅴ** | **审稿后 / 远期** | 按意见 **1 格复现** 或 **补脚注**；远期 **新 `kind`** | **阶段 6** 登记表；**阶段 7** 另立项（**`PROJECT_MASTER_PLAN`**） |

---

## Ⅰ 阶段 5 成文（**默认下一步优先**）

| 顺序 | 动作 | 验证点 | 产出 |
|------|------|--------|------|
| **Ⅰ-1** | 读 **`SUBMISSION_PACK.md` §A1–A2** + **`FIGURE_CAPTIONS_STAGE1.md`** 七轴表 | 能说出 **主验证轴**（树+Mamba+SSGS/M1）与 **副线**（检索/探针） | 心里有「哪些表绝对不能混」 |
| **Ⅰ-2** | 把 **§A3 / §A3b**、**§A2.1** 粘进 LaTeX/Word | 每个 **测量轴** 在正文或附录 **有脚注或一句边界** | 主稿可给导师/合作者通读 |
| **Ⅰ-3** | 全文检索 **`results/`** 引用 | 每个 **basename** ↔ **`EXPERIMENT_REGISTRY`** **一行** | **§A2 核对表** 全打勾 |
| **Ⅰ-4** | 摘要 / 讨论 / 局限 | 不出现 **L4 级 Agent 已证成**；**三风险** 与 **`PHASE1_MANUSCRIPT` §9.2** 一致 | 可投版本叙事 |

---

## Ⅱ 数据冻结与仓库卫生（**可与 Ⅰ 穿插**）

| 顺序 | 动作 | 验证点 |
|------|------|--------|
| **Ⅱ-1** | 仓根 **`aggregate_ssgs_mamba_wikitext_json.py`** + **`aggregate_ssgs_vs_kv_wikitext_json.py`**（见 **`NEXT_EXPERIMENTS_COMMANDS.md` §12**） | **`json_path`** 列为 **`results/metrics_result/...`**；**stdout `N row(s)`** 记下 |
| **Ⅱ-2** | **`python -m pytest tests/ -q`**（**AutoDL / mamba2**） | 全绿；数字回填 **`NEXT_RESEARCH_PLAN` §1**（若变更） |
| **Ⅱ-3** | **`git add` / `commit` / `push`** | **`git status`** 干净；无 **`metrics_result` 手改脏行** |

---

## Ⅲ 可选实验（**M2**；**不阻塞 Ⅰ**）

> **原则**：已有 **M1 n64 + L3（1617Z）**、**path-batch**、**§7**、**SSGS grid** 时，**只做「有叙事收益」的一条** 即可。

| 顺序 | 实验 | 验证什么 | 命令入口 |
|------|------|----------|----------|
| **Ⅲ-1** | **M1 · `chunk_len=12`**（**n8 或 n64** 选一） | **同 harness** 在 **c12** 下 **三臂仍 `ok`**；与 **A2-S2 c12** **脚注分列** | **`SSGS_MAINLINE_M1.md` §6.2 B3** — 直调 **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py --chunk-len 12`** |
| **Ⅲ-2** | **同树 path-batch + SSGS bundle** | **path-batch** 与 **SSGS** **同一建树** 可跑通一行 JSON | **`run_ssgs_mamba_wikitext_cuda.sh`** **`RUN_WIKITEXT_SMOKE=1`**（**`NEXT_EXPERIMENTS_COMMANDS.md` §10**） |
| **Ⅲ-3** | **`git pull` 后 M1 单格** | **`git_sha`** 与当前代码一致 | **`M1_LEAVES="8"`** + 新 **`M1_STAMP`**（**`run_m1_ssgs_vs_kv_wikitext_cuda.sh`**） |

**默认建议**：若时间紧 — **跳过 Ⅲ**，直接 **Ⅰ + Ⅱ**；若导师要 **「c12 对齐」** — 只做 **Ⅲ-1** 一格。

---

## Ⅳ 截稿与投出

- **冻结**：**commit hash**、**主图 PNG 文件名**、**CSV basename** 写入 **方法/附录**。  
- **自检**：**`SUBMISSION_PACK.md` §A8** 过一遍。  
- **不**在截稿前夜改 **`results/metrics_result/*` 文件名**（除非同步改登记册与全文）。

---

## Ⅴ 审稿后（阶段 6）与远期（阶段 7）

| 类型 | 做什么 | 备注 |
|------|--------|------|
| **审稿补实验** | 点名复现、补 **1 个 STAMP**、补表 | 每条意见 → **登记行或正文修改**（**`RESEARCH_PHASES_0_TO_DONE.md` §阶段 6**） |
| **加分项** | **B-S3**、**训练型 L3**、**RAPTOR 式树** | **新 `kind` / 新 harness**；**禁止**与主表无脚注合并 |

---

## 本周「开工」最小序列（可照抄勾选）

1. [ ] **Ⅰ-2**：**§A3** 入稿（或先 **Ⅰ-1** 通读 30 分钟）。  
2. [ ] **Ⅱ-1**：本机/仓根 **重聚合** 两个 **grid CSV**（若刚从云端拷回 JSON）。  
3. [ ] **Ⅱ-2**：下次 **AutoDL** 开机时跑 **pytest** 全量。  
4. [ ] **Ⅲ**：仅在有明确叙事需求时选 **Ⅲ-1** 或 **Ⅲ-2** 一条。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-11 | 初版：Ⅰ–Ⅴ 段 + M2 可选序 + 本周最小勾选 |
