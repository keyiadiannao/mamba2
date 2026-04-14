# 当前迭代（精简执行版）

> 单一执行主线：`docs/overview/planning/MASTER_EXPERIMENT_PLAN_E2E.md`  
> 当周勾选与阻塞：本文件  
> 历史细节：`NEXT_RESEARCH_PLAN.md`、`RESEARCH_STATUS_AND_DIRECTION.md`、`EXPERIMENT_REGISTRY.md`

---

## 当前共识（锁定）

- 主创新定位：`Mamba + SSGS + 树状RAG` 的**导航/检索侧代价结构**，不是生成质量主战场。
- 架构口径：**Navigator-Generator 解耦**（Navigator 负责树内探索/回溯；Generator 负责最终生成）。
- `flat_topk_semantic`：保留为反例/对照臂；**不再做** `k sweep / BM25` 扩展。
- 生成质量 PK：仅作小规模参考项，不作为主结论驱动项。

---

## 本周 P0（必须完成）

- [ ] 文稿收敛：把“解耦架构 + 导航侧贡献边界”落到主稿与摘要（与 `SUBMISSION_PACK.md` 对齐）。
- [ ] 主表核对：`SUBMISSION_PACK §A2` 路径与 `EXPERIMENT_REGISTRY.md` 逐字一致。
- [ ] 实验入口清理：下一轮只保留主线命令块（M1 / E2E 主线），避免语义检索支线命令继续扩张。
- [ ] 推送前检查：`py -3 -m pytest tests/ -q`（或等价可用环境）+ `git status` 干净。

---

## 下一轮实验（主线）

### 1) M1 同树 DFS 主线（优先）

- 目标：继续巩固“回溯代价结构”证据链（同协议、同树、同预算维度）。
- 产出：新 `TASK/X` 结果 + `EXPERIMENT_REGISTRY` 一行 + 必要脚注更新。

### 2) E2E 主线对照（次优先）

- 目标：在同协议 reader 下保持主线臂对照稳定，不新增无关支线。
- 约束：如新增 baseline，必须先写清 comparability 与预算口径。

### 3) 生成质量参考（小项）

- 目标：作为补充观察，不影响主线判断。
- 约束：不将其写成“主贡献优劣证明”。

---

## 阻塞项

- 云端算力窗口与预算（48G 排队时延）。
- 本机与云端环境差异（仅同机同协议结果进入主比较）。

---

## 历史记录位置（避免本文件继续膨胀）

- 历史周报与旧任务展开：`docs/overview/execution/NEXT_RESEARCH_PLAN.md`
- 证据层级与七轴边界：`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md`
- 全量实验真值与路径：`docs/experiments/planning/EXPERIMENT_REGISTRY.md`
