# 当前迭代（滚动）

> 每 1–2 周更新一次「周期」与勾选；完成后把结论一行写入 `EXPERIMENT_REGISTRY.md`。

## 周期

**开始**：2026-04-07  
**目标结束**：2026-04-20（约两周）

---

## 本迭代目标

把阶段 1 从**纯合成树**推进到**可读的文本形树 + 同一 reader 基准槽位**，并准备好 **AutoDL** 上扩大扫参或安装 Mamba 的环境说明。

---

## 任务清单

- [x] 总体规划文档 `PROJECT_MASTER_PLAN.md`
- [x] 扫参 CSV 增强：`gpu_name`、`torch_version`；合并多机 CSV 脚本 `scripts/merge_sweep_csv.py`
- [x] **文本形浅树**：样例叶文本 + 自底向上建树 + `benchmark_text_tree.py` + `run_reader_benchmark_on_paths`（确定性嵌入，非神经 encoder）
- [x] **数据约定**：`data/raw/sample/` 8 段合成 `.txt` + `docs/DATASETS.md`；`prepare_leaves_from_corpus.py` 生成叶文件
- [x] **AutoDL 文档**：`docs/AUTODL_SETUP.md` + `SYNC` 索引；**已在 3090 实例跑通** smoke + `sweep_autodl.csv`（见 `EXPERIMENT_REGISTRY`）
- [x] **本地最小 Mamba**：`transformers.MambaModel` 小配置 smoke（无需 `mamba-ssm`），见 `scripts/smoke_mamba_minimal.py`
- [ ] （可选）**mamba-ssm**：在 AutoDL 上安装融合内核；与上述脚本对比速度或换更大 checkpoint
- [x] **树路径三 reader**：`Mamba2PathReader` 接入 `benchmark_core` / `sweep` / `benchmark_text_tree`（默认开启，`--no-mamba2` 可关）

---

## 阻塞项

- （填写：无 / 依赖项）

---

## 上迭代归档（简述）

- 环境 `mamba2`、cu128、玩具树基准、本地 preset 扫参 8 点 CSV。
