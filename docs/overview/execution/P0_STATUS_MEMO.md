# P0 状态备忘录（§Ⅸ 实证冻结 · 2026-04）

> **用途**：导师/合作者对齐 **「已有什么证据、不能混什么表、下一步问什么」**；**单一细节与 basename** 仍以 **`EXPERIMENT_REGISTRY.md`** 为准。

---

## 1. 工程与机制（门闩 §Ⅷ）

- **G1–G5**：**`ENGINEERING_NORTH_STAR_PLAN.md`**；统一 CLI **`scripts/engineering/run_engineering.py`**；**CI** **`engineering-tests`**（与本地 **`pytest`** 工程子集一致）。
- **分项表**：**path-batch 玩具**、**G3 预训因果 LM**、**SSGS/M1** —— **须脚注分列**，见 **`ENGINEERING_NORTH_STAR_PLAN.md` §4.1 / §4.3**。

---

## 2. TASK 实证快照（Wikitext · 战略 B）

| 块 | id（登记册） | 一句结论 |
|----|----------------|----------|
| **Sprint 1（b）** | **`TASK-…-sprint1b-…`** | **root_child·stratified·n32**：ridge **近饱和**；**Table B**（SSGS/M1）为 **导航代价**，**与 Table A 不可数值比较**。 |
| **1c-A** | **`TASK-…-1cA-…`** | **sibling·leaf_heldout·n32·H=8**：**test 28 对**，reader **有区分**；**`init_seed` 有方差**；**baseline_raw 0.857 恒定**（与实现一致）。 |
| **1c-B** | **`TASK-…-1cB-…`** | **n8 sibling·stratified**：**test 仅 7 对**，四臂 **0.857·std0**，**浅档占位**；**与 1c-A 分列**。 |
| **几何** | — | **`root_child` + `leaf_heldout` @ n=32** 在当前 **前缀/后缀叶划分** 下 **不可行**（**`PLAN_NOW_TO_DONE.md` §Ⅸ-4**）。 |

---

## 3. 禁止混读（写进任何对外材料前自检）

1. **Table A**（**`task_wikitext_path_pair` · `ridge_concat.test_acc`**）与 **Table B**（**SSGS/M1 · 墙钟/快照/峰值**）—— **任务目标不同**。  
2. **（b）root_child·stratified** 与 **（1c-A）sibling·leaf_heldout** —— **cohort 与划分不同**，**非**同一难度。  
3. **path-batch 主表**（**wall-clock / naive·fused**）与 **TASK / ENG** —— **须列名 + 脚注**。  
4. **G3 预训因果** 与 **玩具 path-batch** —— **§Ⅷ / G5** 已要求 **分表**。

---

## 4. 已支持 / 未支持的叙事（避免 oversell）

| 可写 | 尚不可作为主结论 |
|------|------------------|
| 同树 **SSGS/M1** 可复现 **代价迹**；**ENG** 闭环 | **「回溯提高同一监督下的 test acc」** |
| **1c-A** 显示 **held-out + 非饱和** 与 **reader/init 方差** | **「Mamba 在下游全面优于 TF」**（ridge 上 **非恒定**） |
| **动机层**：**`MOTIVATION_MAMBA_TREE_RAG_SSGS_REFERENCE.md`** | **RAPTOR 数字** 照抄进本文 **不作原论文复现标注** |

---

## 5. Sprint 2（a）与「树 RAG vs 平面 RAG」

- **Sprint 2（a）目标**（**`PLAN_NOW_TO_DONE.md` §Ⅸ-6**）：在 **同一任务协议** 下，**SSGS/回溯路径表示** 进入 **与当前 ridge 可比的头**，产出 **可登记 `kind` + `TASK-*`**，回答 **「回溯是否提升 acc」**（或诚实负面结果）。  
- **是否必须先做树 vs 平面？**  
  - **不必阻塞 2（a）**：核心缺口是 **统一监督下的 acc**，可在 **树索引 + 叶对任务** 上先闭环。  
  - **树 vs 平面** 是 **§Ⅸ 另一维度**（**`PLAN` §Ⅸ-1**），对应 **`PLAN` P2 / 平面 F0**（**`benchmark_wikitext_tree` 尚无 `--flat`**，需 **小 PR**）。建议 **2（a）与平面基线并行规划**；**论文级「树状优势」** 叙事 **建议** 在 **同语料、同预算** 下补 **平面臂**，**非** Sprint 2（a）单点完成的必要条件。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-07 | 初版：P0 冻结 + Sprint 2（a）与树/平面关系 |
