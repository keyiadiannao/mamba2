# 当前迭代（滚动）

> 每 1–2 周更新一次「周期」与勾选；完成后把结论一行写入 `docs/experiments/EXPERIMENT_REGISTRY.md`。

## 周期

**开始**：2026-04-07  
**目标结束**：2026-04-20（约两周）

---

## 本迭代目标

把阶段 1 从**纯合成树**推进到**可读的文本形树 + 同一 reader 基准槽位**，并准备好 **AutoDL** 上扩大扫参或安装 Mamba 的环境说明。

---

## 任务清单

- [x] 总体规划文档 `docs/overview/PROJECT_MASTER_PLAN.md`
- [x] 扫参 CSV 增强：`gpu_name`、`torch_version`；合并多机 CSV 脚本 `scripts/benchmarks/merge_sweep_csv.py`
- [x] **文本形浅树**：样例叶文本 + 自底向上建树 + `scripts/benchmarks/benchmark_text_tree.py` + `run_reader_benchmark_on_paths`（确定性嵌入，非神经 encoder）
- [x] **数据约定**：`data/raw/sample/` 8 段合成 `.txt` + `docs/experiments/DATASETS.md`；`scripts/data/prepare_leaves_from_corpus.py` 生成叶文件
- [x] **AutoDL 文档**：`docs/environment/AUTODL_SETUP.md` + `SYNC` 索引；**已在 3090 实例跑通** smoke + `sweep_autodl.csv`（见 `EXPERIMENT_REGISTRY`）
- [x] **本地最小 Mamba**：`transformers.MambaModel` 小配置 smoke（无需 `mamba-ssm`），见 `scripts/smoke/smoke_mamba_minimal.py`
- [ ] （可选）**mamba-ssm**：在 AutoDL 上安装融合内核；与上述脚本对比速度或换更大 checkpoint
- [x] **树路径三 reader**：`Mamba2PathReader` 接入 `benchmark_core` / `sweep` / `benchmark_text_tree`（默认开启，`--no-mamba2` 可关）
- [x] **公开语料浅树**：Wikitext-2（HF `datasets`）→ `hf_corpus.wikitext2_leaf_chunks` → `scripts/benchmarks/benchmark_wikitext_tree.py`（与合成叶同一 reader 槽位）

---

## 阻塞项

- **云端算力**：暂无空闲 AutoDL/服务器 → **融合内核扫参、大叶数、检索头训练** 顺延；见下文「无服务器阶段」优先事项。

---

## 无服务器阶段（优先事项，与讨论对齐）

在只有 **本机 5060** 的前提下，仍与总目标一致：**树内同 harness、叙事上预留 SSGS / 检索头**，但不假装已有 48G 上的数字。

| 优先级 | 方向 | 可执行项 |
|--------|------|----------|
| P0 | **阶段 1 收束** | 本地扫参 CSV + **`scripts/benchmarks/plot_tree_reader_sweep.py`** 出图（例：`results/metrics/figures/sweep_readers_20260410_local5060.png`）；登记见 `EXPERIMENT_REGISTRY` |
| P0 | **真实语料线** | 继续用 `scripts/benchmarks/benchmark_wikitext_tree.py`（及合成叶）巩固「非玩具文本 + 同 reader」；可选：把一次完整 JSON 结果贴进 `experiments/A-20260410-wikitext-shallow-tree/` |
| P1 | **叙事与可检验命题** | `docs/research/RESEARCH_NOTES.md` **§7 测量协议草案**（Mamba 快照内容、TF-R1/TF-KV 基线、报表字段） |
| P1 | **SSGS 最小原型** | `src/rag_tree/ssgs.py` + `TreeNode.state_snapshot` + `tests/test_ssgs.py`；**不接真实 LM** |
| P2 | **检索头** | 读论文与接口草图；代码保持占位，等 GPU 再做探针 |
| 延后 | **mamba-ssm 融合**、大叶数、`--max-leaves` 大网格 | **有服务器再开** |

**讨论结论（写入此表的目的）**：主故事仍是 **`docs/overview/PROJECT_MASTER_PLAN.md` §1.1 树内 Mamba vs Transformer**；**状态快照 / SSGS** 作为协议层贡献，需 **公平基线下的曲线** 支撑，避免仅停留在类比。

---

## 上迭代归档（简述）

- 环境 `mamba2`、cu128、玩具树基准、本地 preset 扫参 8 点 CSV。
