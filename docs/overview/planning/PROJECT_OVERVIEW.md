# 项目总览与推进计划

> **文档导航**：分层索引与「先读谁」以 **`docs/README.md`** 为准。本文保留 **目标、A/B/C/X 分类、仓库目录树、阶段全景**；**可执行周任务**以 **`docs/overview/execution/CURRENT_SPRINT.md`** 为准，**实验登记**以 **`docs/experiments/planning/EXPERIMENT_REGISTRY.md`** 为准（不必与 **`ROADMAP.md`** 重复勾选同一批项）。

---

## 1. 研究目标（一句话）

在**树状结构化 RAG**场景下，以 **Mamba-2** 为核心 reader，结合**检索头机制**，验证**效率（延迟/显存）**与**效果（导航/问答）**，并探索**状态快照回溯**等系统级优势。

---

## 2. 知识分类（三块并列、可独立交付）

| 代号 | 主题 | 产出形态 | 与目录对应 |
|------|------|----------|------------|
| **A** | Mamba-RAPTOR / 树状 RAG 框架 | 流水线代码、树构建与遍历、基线对比 | `src/rag_tree/`, `experiments/A-*` |
| **B** | 检索头：发现与分析 | 探针、可视化、与层/头的关联 | `src/retrieval_head/`, `experiments/B-*` |
| **C** | 检索头：轻量注入与训练 | 模块、训练脚本、消融 | `src/retrieval_head/`, `experiments/C-*` |
| **X** | 横切：回溯/快照/评测协议 | 工具与统一指标；**SSGS**（含 **`MambaNavState` / `dfs_ssgs_mamba`**）与 §7 玩具 JSON | `src/rag_tree/ssgs.py`, `src/rag_tree/mamba_cache_utils.py`, `experiments/X-*` |

**四条线依赖与范围长表**：**`PROJECT_MASTER_PLAN.md` §2–§3**（本节仅速查，避免双写）。

**验证实验优先顺序（建议）**

1. **X + A 最小闭环**：**同一棵树、固定检索预算**，**Mamba-2 vs Transformer reader**（主对照）；**平面 top-k RAG** 仅作可选消融，用于说明「树索引」相对「扁平块检索」的收益，**不必**与主对照混成一条必须故事线。
2. **B**：在固定模型与数据子集上做检索相关行为的分析（证明「头/机制是否存在可写点」）。
3. **C**：在 48G 上小规模注入训练，与 A 结合做端到端消融。

---

## 3. 仓库目录结构（约定）

```
mamba2/
├── docs/                    # 见 docs/README.md
│   ├── overview/
│   │   ├── planning/        # 总体规划：MASTER_PLAN、RESEARCH_STATUS、PROJECT_OVERVIEW、ROADMAP
│   │   └── execution/       # 实施：NEXT_RESEARCH_PLAN、CURRENT_SPRINT、SUBMISSION_PACK
│   ├── experiments/
│   │   ├── planning/        # 登记册、DATASETS、PHASE1_VALIDATION_PLAN
│   │   └── phases/          # 阶段成稿：PHASE*、FIGURE_CAPTIONS、COMPLETE_SUMMARY
│   ├── environment/
│   │   ├── runbooks/        # 执行手册：5060、AutoDL、扫参、同步、安装步骤
│   │   └── troubleshooting/ # 排障：CRLF、Git 合并等
│   └── research/            # 研究笔记（如 SSGS）
├── configs/                 # 实验 YAML（超参、路径占位符，不含密钥）
├── src/                     # 可复用库代码（按 A/B/C/X 分子包）
├── experiments/             # 每次实验一个子目录：config + 笔记 + 指向 results 的 id
├── scripts/                 # 见 scripts/README.md
│   ├── smoke/               # 环境冒烟、Mamba 最小前向
│   ├── benchmarks/        # 树基准、扫参、CSV 合并
│   ├── data/                # 叶文件准备
│   └── sync/                # 双机同步示例
├── data/                    # 仅占位；真实数据见 SYNC 文档（本地/服务器路径）
├── checkpoints/             # 仅占位；大文件不提交 Git
└── results/                 # 指标、小图、tensorboard 等；大体积子目录可 gitignore
```

**实验子目录命名**：`experiments/{A|B|C|X}-{YYYYMMDD}-{短描述}/`，内含 `README.md`（假设、命令、结论一行）。

---

## 4. 阶段计划（与 6 个月总目标对齐，可裁剪）

**月级阶段表、周次重叠说明**：**`PROJECT_MASTER_PLAN.md` §4**。**阶段 2 任务里程碑（A2-S*）**：**`NEXT_RESEARCH_PLAN.md` §2**。本节不复制，避免与总体规划双写。

---

## 5. 文档维护规则

- **改代码必改登记**：新实验在 `docs/experiments/planning/EXPERIMENT_REGISTRY.md` 增一行；重要结论复制到对应 `experiments/.../README.md`。
- **周度（可执行勾选）**：**`docs/overview/execution/CURRENT_SPRINT.md`**。**周历模板 / 历史**：**`docs/overview/planning/ROADMAP.md`**（勿与 sprint 重复维护同一批勾选项）。
- **月度**：在 **`PROJECT_MASTER_PLAN.md`** 或 **`RESEARCH_STATUS_AND_DIRECTION.md`** 补一句「本月结论与下月假设」，或写入 sprint 顶注。

---

## 6. 相关文件索引

**完整分层索引与单一权威矩阵**：**`docs/README.md`**。
