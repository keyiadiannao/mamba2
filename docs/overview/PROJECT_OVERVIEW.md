# 项目总览与推进计划

> **文档导航**：分层索引与「先读谁」以 **`docs/README.md`** 为准。本文保留 **目标、A/B/C/X 分类、仓库目录树、阶段全景**；**可执行周任务**以 **`docs/overview/CURRENT_SPRINT.md`** 为准，**实验登记**以 **`docs/experiments/EXPERIMENT_REGISTRY.md`** 为准（不必与 **`ROADMAP.md`** 重复勾选同一批项）。

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

**验证实验优先顺序（建议）**

1. **X + A 最小闭环**：**同一棵树、固定检索预算**，**Mamba-2 vs Transformer reader**（主对照）；**平面 top-k RAG** 仅作可选消融，用于说明「树索引」相对「扁平块检索」的收益，**不必**与主对照混成一条必须故事线。
2. **B**：在固定模型与数据子集上做检索相关行为的分析（证明「头/机制是否存在可写点」）。
3. **C**：在 48G 上小规模注入训练，与 A 结合做端到端消融。

---

## 3. 仓库目录结构（约定）

```
mamba2/
├── docs/                    # 见 docs/README.md
│   ├── overview/            # 总览、总体规划、ROADMAP、CURRENT_SPRINT
│   ├── experiments/         # EXPERIMENT_REGISTRY、PHASE1/2 成文稿与草稿、DATASETS
│   ├── environment/         # SYNC、AUTODL、mamba-ssm 安装说明
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

### 阶段 0：项目基建（第 1–2 周）

- 双机 Python 环境一致（版本锁定文件见 `environment/`）。
- 跑通「小数据 + 极小模型」在 5060 上的 smoke test；同一脚本在 AutoDL 上只改路径与 batch。
- 填满 `docs/experiments/EXPERIMENT_REGISTRY.md` 表头；第一次登记「环境复现」实验。

### 阶段 1：验证实验 — 树 RAG + Reader 对比（核心优先）

- 构建或接入一层 RAPTOR 式树（可先浅层、小规模语料）。
- 指标：**同等深度/预算**下的质量 + **序列长度扫描**下的延迟/显存。
- 结论写入 registry：是否保留「Mamba 系统叙事」。

### 阶段 2：检索头 — 分析（B）

- 冻结主干，做探针/对比激活；文档化「哪些层/通道与检索决策相关」。

### 阶段 3：检索头 — 注入与联合（C + A）

- 轻量模块 + 训练策略；与树导航决策对齐；消融表。

### 阶段 4：扩展（可选）

- 状态快照回溯协议、生成式节点索引等（与讨论稿一致，依赖阶段 1 是否显示优势）。

---

## 5. 文档维护规则

- **改代码必改登记**：新实验在 `docs/experiments/EXPERIMENT_REGISTRY.md` 增一行；重要结论复制到对应 `experiments/.../README.md`。
- **周度**：更新 `docs/overview/ROADMAP.md` 勾选与下周三条可执行任务。
- **月度**：在本文件「阶段计划」下追加一段「本月结论与下月假设」。

---

## 6. 相关文件索引

| 文档 | 用途 |
|------|------|
| `docs/overview/ROADMAP.md` | 周粒度任务与里程碑 |
| `docs/environment/SYNC_AND_ENVIRONMENTS.md` | 5060 / AutoDL 分工与同步 |
| `docs/experiments/EXPERIMENT_REGISTRY.md` | 实验 ID、命令、指标、结论 |
| `docs/experiments/PHASE1_VALIDATION_PLAN.md` | 阶段 1 验证目标、扫参网格、产出与判据 |
| `docs/overview/PROJECT_MASTER_PLAN.md` | **6 个月级总体规划**、四条线、风险、里程碑 |
| `docs/overview/CURRENT_SPRINT.md` | **当前 1–2 周**执行任务（滚动） |
| `docs/experiments/DATASETS.md` | 数据与样例路径约定 |
| `docs/environment/AUTODL_SETUP.md` | 云端实例克隆、环境、smoke、扫参 |
| `docs/environment/MAMBA_SSM_INSTALL_LINUX.md` | AutoDL 上安装融合核（causal-conv1d、mamba-ssm） |
