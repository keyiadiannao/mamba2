# 下一阶段：记录模板 + 运行指令

> **用途**：**A2-S2（dim128）** 已归档后，按优先级继续 **扩规模 / 对齐 HEAD / 机制探针**。  
> **登记**：每次新网格 **新开 `EXPERIMENT_REGISTRY` 行**（勿与 **`A-stage2-wikitext-grid-v1`** 无说明合并）。  
> **公平性**：**5060 naive** 与 **3090 fused**、**dim128** 与 **dim256** 均须 **分列 + 脚注**（见 **`RESEARCH_STATUS` §6**、**`FIGURE_CAPTIONS`** 五轴）。

---

## 0. 服务器公共前置（每条流水线前先执行）

```bash
cd /root/autodl-tmp/mamba2
git pull origin master

find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_stage2_wikitext_grid.sh scripts/benchmarks/run_server_wikitext_dim256_grid.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

export HF_ENDPOINT=https://hf-mirror.com
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result"

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

## 4. 【本机 / CPU】机制线 **B-S2+**（不占 3090）

与 **path-batch** 分列；见 **`RETRIEVAL_HEAD_NOTES.md` §4 / §7**。

```powershell
cd d:\cursor_try\mamba2
conda activate mamba2

python scripts\research\probe_path_reader_linear.py --cpu --out-json results\metrics\probe_path_reader_linear_text16_heldout_rerun.json
```

（参数可按需加 **`--train-steps`** 等；大模型 / **per-head** 属 **B-S3**，需另排 **48G**。）

---

## 5. 【无命令】成文收口

在 **`PHASE1_MANUSCRIPT.md` §8** 或 **`PHASE2_DRAFT.md`** 增加 **5060 naive 四格** vs **3090 fused dim128（R1/R2）** 对照子表；勿与 **paper_main 合成树** 无标注混表。

---

## 6. 拉回 Windows 并入 Git（示例）

```powershell
cd d:\cursor_try\mamba2
# 将服务器 metrics_result 中新文件拷到 results\metrics_result\
git add results/metrics_result/benchmark_wikitext_stage2_dim256_*.json results/metrics_result/benchmark_wikitext_stage2_dim256_*.csv results/metrics_result/benchmark_wikitext_stage2_dim256_*.txt
# 编辑 docs/experiments/EXPERIMENT_REGISTRY.md 后
git commit -m "metrics: Wikitext dim256 fused grid + registry"
git push origin master
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初版：公共前置、HEAD 单格、**dim256** 脚本、n32 可选、B-S2+、成文指针 |
