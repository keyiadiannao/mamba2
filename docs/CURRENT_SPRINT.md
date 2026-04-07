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
- [ ] **数据约定**：在 `data/raw/sample/` 放 3–10 篇短文（或下载脚本说明），`docs/DATASETS.md` 写清来源与许可
- [ ] **AutoDL**：实例上 `git clone` + 创建 conda + 按 GPU 安装 torch；跑通 `smoke_local.py` 与 `sweep_tree_benchmark.py --preset local`，CSV 合并到本机
- [ ] （可选）**mamba-ssm**：在 AutoDL 上尝试安装；成功则 registry 记一条 import + 微型 forward

---

## 阻塞项

- （填写：无 / 依赖项）

---

## 上迭代归档（简述）

- 环境 `mamba2`、cu128、玩具树基准、本地 preset 扫参 8 点 CSV。
