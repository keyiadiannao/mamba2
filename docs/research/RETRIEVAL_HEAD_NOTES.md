# 检索头：研究笔记（提纲）

> **B-S1 / B-S2**：与「树上读路径」主线并行；不阻塞 **A2** 计时与登记。**B-S2** 探针脚本见 **§4**；文献精读仍可与 **§6** 并行推进。

## 1. 问题陈述

- **目标**：在「长上下文 / 多文档」设定下，模型是否显式或隐式地执行了**检索式**信息路由（例如从大量 token 或块中选出少数相关位置）。
- **与本文主线的关系**：当前仓库的 **path reader** 与 **浅树导航** 是**结构化**读路径；检索头文献讨论的多是 **Transformer 注意力** 或 **专用检索模块**。两者可比的是「**是否**在层/头层面出现可分的检索行为」，而非直接对比 MFU。

## 2. 文献与设计空间（占位）

- **Hidden Attention**、**Retrieval heads**、**RAD** 等：记录论文与代码链接；区分「定义」「探测方法」「任务」。
- **树上读路径**：本文已有 **bottom-up** 建树与 **path batch**；检索头工作多在 **平面序列** 或 **KV 缓存** 上——对照时写明 **不可混读** 的维度（见 **`RESEARCH_STATUS_AND_DIRECTION.md`** §6）。

## 3. 树上导航 vs 平面检索

- **树**：路径由结构约束；读路径长度与 `num_leaves`、fanout 相关。
- **平面检索**：头可能在高熵位置上「跳」到少数关键 span。
- **假设**：若存在「检索式」行为，在 **浅树 + 同 harness** 下可能表现为特定层/头对 **叶块边界** 或 **路径前缀** 的敏感；需单独设计探针（**B-S2**）。

## 4. 探针脚手架（B-S2，已落地）

- **脚本**：`scripts/research/probe_retrieval_correlation.py`  
  - 对 **`sshleifer/tiny-gpt2`**（可换 `--model`）各层 **mean-pool** 隐状态，训练 **岭线性二分类**（NumPy 闭式解，无 sklearn）。  
  - **标签**：`--label-mode marker`（合成子串 **`RETRVPROBE`**）、`digit`（合成插入 **`42`**）、`random`（打乱标签作对照）。  
  - 默认另输出 **`random_label_control`**：与 **marker** 相同文本、**打乱标签** 的每层 test acc（应接近 **0.5**；若 **marker ≫ random**，仅说明「隐状态线性可读」，**不等于** 真实检索头）。  
  - **`--out-json`**：含 `git_sha`、`torch_version`、`layers[].train_acc` / `test_acc`。
- **登记**：**`EXPERIMENT_REGISTRY`** 行 **`X-20260410-retrieval-linear-probe`**（占位，可本地复跑 JSON）。
- **与计时实验分离**：不改变 **A2** harness JSON；探针 JSON 字段独立。

## 5. 明确不做的事

- 不在本阶段将检索头结论写进 **主文定理** 或 **path-batch 公平性** 段落。
- 不把 **5060 vs 3090** 或 **path-batch vs §7** 的混点说成「检索头」问题。

## 6. 下一步（B-S2 深化）

- **文献**：在 §2 填入 Hidden Attention / retrieval heads 等 **引用与一句话方法**。  
- **实验**：在 **path reader 表征**（或 **树上路径文档**）上复用同一探针接口 — 需另开脚本或 `--encoder` 分支时再登记。  
- **头级分析**：当前脚本为 **层向量**；若需 **per-head**，需在 Transformer 层上取 **attention 输出按头切分** 后再探针（工作量 +1）。
