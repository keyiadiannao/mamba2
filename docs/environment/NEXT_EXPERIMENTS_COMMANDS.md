# 下一阶段：记录模板 + 运行指令

> **用途**：**A2-S2（dim128）** 与 **叶数 / dim256 / §7 depth** 已归档后，优先跑 **A2-S3 多种子**（轻负载），可选 **机制探针 B-S2+**。  
> **登记**：每次新网格 **新开 `EXPERIMENT_REGISTRY` 行**（勿与 **`A-stage2-wikitext-grid-v1`** 无说明合并）。  
> **公平性**：**5060 naive** 与 **3090 fused**、**dim128** 与 **dim256** 均须 **分列 + 脚注**（见 **`RESEARCH_STATUS` §6**、**`FIGURE_CAPTIONS`** 五轴）。

---

## 0. 服务器公共前置（每条流水线前先执行）

```bash
cd /root/autodl-tmp/mamba2
git pull origin master   # 若 TLS 失败：本机 pull + PyCharm 同步；见 **SERVER_SWEEP_RUNBOOK** §1

find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_stage2_wikitext_grid.sh \
  scripts/benchmarks/run_server_wikitext_dim256_grid.sh \
  scripts/benchmarks/run_server_wikitext_leavescale.sh \
  scripts/benchmarks/run_server_section7_depth_sweep.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

export HF_ENDPOINT=https://hf-mirror.com
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result" "$MAMBA2_RESULTS_ROOT/metrics"

# 可选：降低 Hub 匿名限速告警
# export HF_TOKEN=hf_你的token
```

（Conda 若在 **`/root/anaconda3`**，请改 **`source`** 路径；代码若不在 **`/root/autodl-tmp/mamba2`**，只改 **`cd`**。）

---

## 1. 【推荐优先】对齐当前 **HEAD**（单格 smoke，约 1 分钟）

确认 JSON 里的 **`git_sha`** 与 **`git rev-parse --short HEAD`** 一致。

```bash
cd /root/autodl-tmp/mamba2
git rev-parse --short HEAD

STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/benchmarks/benchmark_wikitext_tree.py \
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 \
  --warmup 2 --reps 8 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics_result/benchmark_wikitext_headcheck_${STAMP}_n8_c8.json"

grep git_sha "$MAMBA2_RESULTS_ROOT/metrics_result/benchmark_wikitext_headcheck_${STAMP}_n8_c8.json"
```

**跑后**：在 **`EXPERIMENT_REGISTRY`** 用一句话记下 **`STAMP`**、**`git_sha`**、路径（可登记为 **X-*** 辅线，不必动 **A-stage2** 主行）。

---

## 2. 【系统扩展】Wikitext **dim=256** 四格（与 A2-S2 **同拓扑**，新登记）

一键脚本（**fused**、**WARMUP=2** **REPS=8** 默认）：

```bash
cd /root/autodl-tmp/mamba2
export TAG=stage2_dim256
unset STAMP
./scripts/benchmarks/run_server_wikitext_dim256_grid.sh
```

省时（3 格，跳过 **8×12**）：

```bash
export TAG=stage2_dim256
export GRID=minimal
unset STAMP
./scripts/benchmarks/run_server_wikitext_dim256_grid.sh
```

**产出**：**`$MAMBA2_RESULTS_ROOT/metrics_result/benchmark_wikitext_<TAG>_<STAMP>_n*_c*.json`** + **`…_grid_<STAMP>.csv`** + manifest。

**跑后登记（建议）**：在 **`EXPERIMENT_REGISTRY.md`** **新增** id **`A-stage2-wikitext-dim256-v1`**（或自拟），填 **GPU、驱动、`git_sha`、TAG、STAMP、文件路径、一句 Mamba2 峰值趋势**。

---

## 3. 【可选】更重单点：**32 叶** × **chunk=8** × **dim=128**（深度 5）

叶数须为 **`fanout** 的幂**（默认 **fanout=2** → **32** 合法）。比 16 叶 **更吃显存/时间**，先 **单点** 探顶。

```bash
cd /root/autodl-tmp/mamba2
STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/benchmarks/benchmark_wikitext_tree.py \
  --num-leaves 32 --fanout 2 --chunk-len 8 --dim 128 \
  --warmup 2 --reps 8 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics_result/benchmark_wikitext_fused_n32_c8_${STAMP}.json"
```

**注意**：文件名 **无** **`n32_c8`** 中间下划线时，**`aggregate_wikitext_tree_json_grid.py`** 可能无法解析格点；请保持 **`…_n32_c8_…json`** 命名模式（可把 **`_n32_c8`** 放进文件名任意位置）。

---

## 4. 【系统扩展】叶数扫描 **{8,16,32,64}** × **chunk=8** × **dim=128**（**path-batch**）

**目的**：看 **m2_peak / per_step** 随 **叶数** 变化；**TF** 为 **整段 SA O(T²)**，大树时相对 **GRU/Mamba2** 往往更吃亏（见 **`readers.TransformerPathReader`**）。

```bash
cd /root/autodl-tmp/mamba2
export TAG=stage2_leavescale
unset STAMP
./scripts/benchmarks/run_server_wikitext_leavescale.sh
```

**省时 / 降显存**（不要 64 叶）：

```bash
LEAVES="8 16 32" ./scripts/benchmarks/run_server_wikitext_leavescale.sh
```

**产出**：**`benchmark_wikitext_<TAG>_<STAMP>_n*_c8.json`** + **`…_grid_<STAMP>.csv`** + manifest。登记 **A-stage2-wikitext-leavescale-v1**（**`SERVER_SWEEP_RUNBOOK.md` §2f**）。

**128 / 256 叶（可选，建议晚于 8–64）**：**新 `TAG`**（如 **`stage2_leavescale_xl`**），**先 `LEAVES="128"`** 再 **`256`**；语料不够时用 **`CHARS_PER_LEAF=400`**。见 **`SERVER_SWEEP_RUNBOOK` §2f** 末段。

---

## 5. 【§7 玩具协议】**depth 5–6** 扩展（**S1–S4**，与 path-batch **分列**）

**推荐**：整段复制 **`docs/environment/RUN_AUTOADL_SECTION7_NOW.md`**（含 **`sed`/`conda`/`MAMBA2_RESULTS_ROOT`**）。

```bash
cd /root/autodl-tmp/mamba2
unset TAG
unset STAMP
./scripts/benchmarks/run_server_section7_depth_sweep.sh
```

**含 depth=4 对照**（与 **20260421** 归档同深度再跑）：

```bash
DEPTHS="4 5 6" ./scripts/benchmarks/run_server_section7_depth_sweep.sh
```

登记 **X-section7-depth-extension-v1**；详见 **`SERVER_SWEEP_RUNBOOK.md` §2g**。

---

## 6. 【本机 / CPU】机制线 **B-S2+**（不占 3090）

与 **path-batch** 分列；见 **`RETRIEVAL_HEAD_NOTES.md` §4 / §7**。

```powershell
cd d:\cursor_try\mamba2
conda activate mamba2

python scripts\research\probe_path_reader_linear.py --cpu --out-json results\metrics\probe_path_reader_linear_text16_heldout_rerun.json
```

（参数可按需加 **`--train-steps`** 等；大模型 / **per-head** 属 **B-S3**，需另排 **48G**。）

---

## 7. 【无命令】成文收口

在 **`PHASE1_MANUSCRIPT.md` §8** 或 **`PHASE2_DRAFT.md`** 增加 **5060 naive 四格** vs **3090 fused dim128（R1/R2）** 对照子表；勿与 **paper_main 合成树** 无标注混表。

---

## 8. 拉回 Windows 并入 Git（示例）

```powershell
cd d:\cursor_try\mamba2
# 将服务器 metrics_result 中新文件拷到 results\metrics_result\
git add results/metrics_result/benchmark_wikitext_stage2_dim256_*.json results/metrics_result/benchmark_wikitext_stage2_dim256_*.csv results/metrics_result/benchmark_wikitext_stage2_dim256_*.txt
# 编辑 docs/experiments/EXPERIMENT_REGISTRY.md 后
git commit -m "metrics: Wikitext dim256 fused grid + registry"
git push origin master
```

---

## 9. 【优先】**A2-S3** 多种子 + **leaf_heldout**（**3090 fused**，与 path-batch **分列**）

与 **`PHASE1_MANUSCRIPT.md` §10** 第 1 条一致：**`split-seed`** 扫 **5–10** 个，固定 **`--cohort sibling`**（或另开 **`root_child`** 登记），**`--pair-split leaf_heldout --heldout-leaves 6`**（**16 叶** 时 test 叶对 **C(6,2)=15**；**32 叶** 更稳）。

**前置**：**§0**（**`conda activate mamba2`**、**`MAMBA2_RESULTS_ROOT`**、**`HF_ENDPOINT`**）。若报 **`causal_conv1d` … strides … multiples of 8**：先 **`git pull`** — `Mamba2PathReader` 默认已改为 **fused 友好** 的 **num_heads≥8** 拆分（见 **`src/rag_tree/readers.py`**）。

**16 叶 × chunk8 × dim128**（示例）：

```bash
cd /root/autodl-tmp/mamba2
STAMP=$(date -u +%Y%m%dT%H%MZ)
for S in 0 1 2 3 4; do
  python scripts/research/task_wikitext_path_pair.py \
    --num-leaves 16 --fanout 2 --chunk-len 8 --dim 128 \
    --cohort sibling --pair-split leaf_heldout --heldout-leaves 6 --split-seed "$S" \
    --out-json "$MAMBA2_RESULTS_ROOT/metrics/task_wikitext_sibling16_c8_leafheldout6_splitseed${S}_${STAMP}.json"
done
```

**32 叶**（把 **`--num-leaves 32`**，文件名里 **`sibling32`**；**heldout 6** 仍合法）。

**跑后**：汇总各 JSON 的 **`ridge_concat.*.test_acc`**（及 **`raw_concat`** 若需要），在 **`EXPERIMENT_REGISTRY.md`** **新开一行**（勿并入 **A-stage2-wikitext-grid-v1**）；成文与 **path-batch** **分列**（见 **`FIGURE_CAPTIONS_STAGE1.md`** 五轴）。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初版：公共前置、HEAD 单格、**dim256** 脚本、n32 可选、B-S2+、成文指针 |
| 2026-04-09 | **§9**：**A2-S3** 多种子 + **leaf_heldout** 服务器命令；**§0** 增 **`metrics/`**；用途段补 **A2-S3** |
| 2026-04-09 | **已跑通**：**dim256** **`STAMP=20260409T1137Z`**；**n32** 单点；**headcheck** **`20260409T1135Z`** — 见 **`EXPERIMENT_REGISTRY`** |
| 2026-04-07 | **叶数扫描** **`run_server_wikitext_leavescale.sh`**、**§7 depth 5–6** **`run_server_section7_depth_sweep.sh`**；**`SERVER_SWEEP_RUNBOOK` §2f–§2g**；本节重编号 §4–§8 |
| 2026-04-09 | **已跑通叶数扫描**：**`TAG=stage2_leavescale`** **`STAMP=20260409T1257Z`** → **`benchmark_wikitext_stage2_leavescale_*`**；登记 **A-stage2-wikitext-leavescale-v1**；**`SERVER_SWEEP_RUNBOOK` §1** 补 **GitHub TLS / PyCharm** |
| 2026-04-09 | **§4** 末：**128/256 叶** 建议；**`run_server_wikitext_leavescale.sh`** 支持 **`CHARS_PER_LEAF`**；**`SERVER_SWEEP_RUNBOOK` §2f** 大叶数段 |
| 2026-04-09 | **已跑通 XL**：**`TAG=stage2_leavescale_xl`** **`STAMP=20260409T1322Z`（n128）**、**`20260409T1324Z`（n256）**；**`benchmark_wikitext_stage2_leavescale_xl_grid_n128_n256_combined.csv`**；**A-stage2-wikitext-leavescale-xl-v1** |
