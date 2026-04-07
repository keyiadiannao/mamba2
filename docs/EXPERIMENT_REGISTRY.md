# 实验登记册

> 每条实验一行（或一个小节）。**必填**：id、日期、机器、git commit、目的、关键命令、主要指标、一句话结论。

---

## 登记表

| id | 日期 | 机器 | commit | 方向 | 目的 | 关键指标 | 结论 |
|----|------|------|--------|------|------|----------|------|
| env-001 | | 5060+ADL | | X | 环境复现与 smoke | import OK | |
| X-20260407-smoke-local | 2026-04-07 | 5060 | | X | conda env mamba2 冒烟 | torch 2.11.0+cu128, CUDA OK, 50x fwd ~0.27s GPU; mamba_ssm 未装 | OK |
| A-20260407-toy-tree-reader-bench | 2026-04-07 | 5060 | | A | 玩具树 Reader 微基准 | `python scripts/benchmark_tree_walk.py` | TF vs GRU  harness 就绪；待换 Mamba |

---

## 字段说明

- **id**：与 `experiments/` 目录名后缀一致，如 `A-20260407-baseline`。
- **commit**：该次实验所用代码提交；若脏工作区，注明 `dirty` 并简述差异。
- **关键命令**：可粘贴完整一行，或指向 `experiments/.../README.md`。

---

## 待跑实验队列（ backlog ）

| 优先级 | id | 说明 |
|--------|-----|------|
| P0 | | |
| P1 | | |
