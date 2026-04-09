# 检索头：研究笔记（提纲）

> **B-S1 / B-S2**：与「树上读路径」主线并行；不阻塞 **A2** 计时与登记。**B-S2** 探针脚本见 **§4**；文献精读仍可与 **§6** 并行推进。

## 1. 问题陈述

- **目标**：在「长上下文 / 多文档」设定下，模型是否显式或隐式地执行了**检索式**信息路由（例如从大量 token 或块中选出少数相关位置）。
- **与本文主线的关系**：当前仓库的 **path reader** 与 **浅树导航** 是**结构化**读路径；检索头文献讨论的多是 **Transformer 注意力** 或 **专用检索模块**。两者可比的是「**是否**在层/头层面出现可分的检索行为」，而非直接对比 MFU。

## 2. 文献与设计空间（精读入口）

下列条目区分 **对象**（模型里是什么）、**方法**（如何找/如何用）、**任务**（在什么基准上说话）。**完整引用格式**以各文官方页为准；此处给 **稳定链接** 与 **一句话** 便于回到主线对照。

| 线索 | 代表工作 | 对象 / 方法 / 任务（极简） | 与本文 path-batch / 浅树 harness 的关系 |
|------|----------|-----------------------------|----------------------------------------|
| **Retrieval heads** | [arXiv:2404.15574](https://arxiv.org/abs/2404.15574) *Retrieval Head Mechanistically Explains Long-Context Factuality*（Yao Fu 等） | **对象**：少数 attention heads 负责从**长上下文任意位置**拉回信息；**方法**：识别 + 剪枝/掩码因果实验；**任务**：长上下文事实性、CoT 等 | 讨论的是 **全序列自注意力** 内的头；**不是** 本仓库的 **path reader 三对比** 或 **Mamba 状态**。可比的是「内部是否存在可定位的 **信息路由** 单元」。 |
| **Hidden attention sinks** | [arXiv:2406.15765](https://arxiv.org/abs/2406.15765) *Unveiling and Harnessing Hidden Attention Sinks*（Yu 等，ICML 2024） | **对象**：除 BOS 外序列内部也会出现 **attention sink**；**方法**：推理期 **ACT** 校准注意力分布；**任务**：多基准准确率 | 侧重 **注意力质量/校准**，与「检索头」部分重叠话题（**谁吸走概率质量**），但**不**等价于 2404.15574 的 *retrieval head* 定义。 |
| **Induction heads** | [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) *In-context Learning and Induction Heads*（Olsson 等） | **对象**：两层机制实现 **\[A]\[B]…\[A]→\[B]** 式复制；**方法**：电路/注意力图解释；**任务**：ICL | 经典 **平面 token 序列** 上的头部角色；可作「**结构化读路径** 上是否出现类似电路」的**类比参照**，**不可**直接当实验结论。 |
| **RAD（解码期检索）** | [arXiv:2508.02184](https://arxiv.org/abs/2508.02184) *Retrieval-augmented Decoding for Improving Truthfulness…*（Nguyen 等） | **对象**：解码时每步 **检索 grounding 空间** 中的上下文嵌入；**方法**：对 next-token logits 做 **检索式加权**；**任务**：开放生成事实性 | **系统层解码干预**，非「数 MHA 里有几个头」。与 **B-S2 线性探针** 不同轴；若写相关工作需 **分项**（内部头 vs 外部检索增强解码）。 |
| **DuoAttention（工程线）** | [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads* | **对象**：头分为 **retrieval-like** 与 **streaming**；**方法**：差异化 KV 缓存；**任务**：长上下文推理效率 | **KV / 推理系统** 与本文 §7 **TF-KV** 话题更近；与 **path-batch 延迟表** **不可混表**。 |

**树上读路径（本文已有）**：**bottom-up** 建树 + **path batch** + 三 reader；文献上表多在 **平面序列** 或 **KV 策略** 上工作——对照时遵守 **`RESEARCH_STATUS_AND_DIRECTION.md`** §6（**path-batch vs §7**、**5060 vs 3090** 等 **不可混读**）。

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

- **文献**：§2 已列 **入口引用**；精读时把每篇的 **探测/识别算法** 记到本节附录式 bullet（可选新开 `RETRIEVAL_HEAD_NOTES_APPENDIX.md`，避免主文件过长）。  
- **实验**：在 **path reader 表征**（或 **树上路径文档**）上复用 **`probe_retrieval_correlation.py` 的岭探针思路**；建议先用 **`--model gpt2`**（或与本机显存匹配的因果 LM）重跑，替换 **`tiny-gpt2` 玩具维** 的示例 JSON。  
- **头级分析**：当前脚本为 **层 mean-pool**；若对齐 2404.15574 风格，需在 Transformer 内取 **per-head** 统计量再探针（工作量 +1，**B-S3** 级）。  
- **主线并行**：**A2-S2**（Wikitext 小网格，**3090 fused**）仍按 **`NEXT_RESEARCH_PLAN`**，与探针 **独立登记**。
