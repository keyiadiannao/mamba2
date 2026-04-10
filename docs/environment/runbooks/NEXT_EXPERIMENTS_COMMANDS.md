# 下一阶段：记录模板 + 运行指令

> **用途**：**A2-S2（dim128）** 与 **叶数 / dim256 / §7 depth** 已归档后，优先跑 **A2-S3 多种子**（轻负载），可选 **机制探针 B-S2+**。按证据层级在服务器推进时，先看 **§0.5**（**L1–L4** 与块 **A–F**）。  
> **登记**：每次新网格 **新开 `EXPERIMENT_REGISTRY` 行**（勿与 **`A-stage2-wikitext-grid-v1`** 无说明合并）。  
> **公平性**：**5060 naive** 与 **3090 fused**、**dim128** 与 **dim256** 均须 **分列 + 脚注**（见 **`RESEARCH_STATUS` §6**、**`FIGURE_CAPTIONS`** **七轴**）。

---

## 0. 服务器公共前置（每条流水线前先执行）

**一键（推荐，实例重启后先跑）**：仓库根执行

```bash
cd /root/autodl-tmp/mamba2
git pull origin master   # 若失败：本机 pull + 同步；见 **SERVER_SWEEP_RUNBOOK** §1
bash scripts/server/bootstrap_autodl.sh
```

其内部会 **`source`** **`scripts/server/_autodl_env.sh`**（**`HF_ENDPOINT=https://hf-mirror.com`**、**`MAMBA2_RESULTS_ROOT`**、`conda activate mamba2`），并做 **Wikitext `datasets`** 与 **CUDA** 检查。

**手工等价**（与旧版一致）：

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

（Conda 若在 **`/root/anaconda3`**，请改 **`source`** 路径或 **`export CONDA_SH=/root/anaconda3/etc/profile.d/conda.sh`** 后再跑 **`bootstrap_autodl.sh`**；代码若不在 **`/root/autodl-tmp/mamba2`**，**`export MAMBA2_REPO_ROOT=...`**。）

---

## 0.5 按研究证据层级（**L1–L4**）· 服务器推荐序

> **层级定义与边界**：**`docs/overview/planning/RESEARCH_STATUS_AND_DIRECTION.md` §3.5**（**L3** 当前 harness 为 **toy TF-KV + 固定叶头 CE**，**≠** 全量 LM / 可训子头；**L4** 本文不列可跑块）。  
> **跑后**：把 **`$MAMBA2_RESULTS_ROOT`** 下 **JSON** 拷入仓库 **`results/metrics_result/`**（或 **`results/metrics/`**），本机再跑对应 **`aggregate_*.py`**，**`EXPERIMENT_REGISTRY`** 新开行；云路径 **`/root/...`** 勿当仓库真值写进 **CSV**。

| 层级 | 推进目标（摘要） | 服务器动作（先 **§0** `bootstrap`） |
|------|------------------|-------------------------------------|
| **L1** | 结构正确、**`git_sha`** 对齐 | **下面块 A**（与 **§1** 同义） |
| **L2** | 容量 / 成本：同树 **SSGS vs TF-KV**、**path-batch**、辅线 **SSGS 叶扩** | **块 B**（M1 **n64**）、**块 C**（**§4** 叶扫脚本）、**块 D**（SSGS **n128**） |
| **L3** | 机制探针（**最小**）：隐状态 / 下游 CE | **块 E**（M1 **`M1_WITH_L3_*`**）；**B-S2+ CUDA** 见 **块 F** |
| **L4** | 训练闭环等 | 见 **`RESEARCH_STATUS` §3.5**，暂无与本仓库 **§4** 同级的单脚本块 |

**块 A — L1：`HEAD` 单格 smoke**

```bash
cd /root/autodl-tmp/mamba2
git pull origin master && bash scripts/server/bootstrap_autodl.sh

STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/benchmarks/benchmark_wikitext_tree.py \
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 \
  --warmup 2 --reps 8 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics_result/benchmark_wikitext_headcheck_${STAMP}_n8_c8.json"
```

**块 B — L2：M1 扩到 **64 叶**（**fanout=2** 合法）**

```bash
cd /root/autodl-tmp/mamba2
M1_LEAVES="64" bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
# 与已有 CSV 合并行：AGGREGATE_APPEND=1
```

**块 C — L2：path-batch 叶数扫描**（与 **§4** 一致）

```bash
cd /root/autodl-tmp/mamba2
export TAG=stage2_leavescale
unset STAMP
./scripts/benchmarks/run_server_wikitext_leavescale.sh
```

**块 D — L2 辅线：SSGS **n128**（**c8 dim128**）+ 汇总**

```bash
cd /root/autodl-tmp/mamba2
EXTRA_LEAVES="128" AGGREGATE_APPEND=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
```

**块 E — L3（最小）：M1 + 隐状态 / 下游 CE**（JSON 变大；可先 **n8** 或 **n64** 单点）

```bash
cd /root/autodl-tmp/mamba2
# 隐状态（每步 trunk 状态 vs 金路径）
M1_LEAVES="8" M1_WITH_L3=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
# 或：下游固定叶头 CE（与「树 LM 可学习头」分列叙述，见 **`RESEARCH_STATUS` §3.5**）
M1_LEAVES="8 16 32 64" M1_WITH_L3_DOWNSTREAM_CE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
```

**块 F — L3 辅线：B-S2+ **CUDA**（与 **`LOCAL_5060_RUNBOOK`** CPU 分列）**

```bash
cd /root/autodl-tmp/mamba2
STAMP=$(date -u +%Y%m%dT%H%MZ)
# 建议写入 **metrics_result**，与 M1 / path-batch 同目录，避免只同步 **metrics_result/** 时漏拷 **metrics/**
OUT="$MAMBA2_RESULTS_ROOT/metrics_result/probe_path_reader_linear_text16_heldout_train50_cuda_${STAMP}.json"
mkdir -p "$(dirname "$OUT")"
# 去掉 --cpu → 有 CUDA 时走 GPU（与 **`probe_path_reader_linear.py`** 默认一致）
python scripts/research/probe_path_reader_linear.py \
  --n-leaves 16 --leaf-split heldout --train-steps 50 --train-lr 3e-3 \
  --out-json "$OUT"
echo "wrote $OUT"
```

（**常见失误**：只复制 **`python … --out-json "$OUT"`** 而未在同一 shell 先执行 **`STAMP=…`** 与 **`OUT=…`**，会得到 **`probe_…_cuda_.json`**。）

**块 G — 阶段 C：L3 轨迹甲·乙（**`tf_kv_trajectory_l3_minimal`**；**≠ M1**、**≠ path-batch**）**

```bash
cd /root/autodl-tmp/mamba2
STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/research/benchmark_tf_kv_trajectory_l3_minimal.py \
  --device cuda \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics_result/tf_kv_trajectory_l3_minimal_cuda_${STAMP}.json"
```

本机 **CPU**（**须 torch 可加载**）：**`--device cpu`**。登记 **X-20260411-tf-kv-trajectory-l3-minimal**；**`pytest tests/test_tf_kv_trajectory_l3.py`**。

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

**推荐**：整段复制 **`docs/environment/runbooks/RUN_AUTOADL_SECTION7_NOW.md`**（含 **`sed`/`conda`/`MAMBA2_RESULTS_ROOT`**）。

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
# 编辑 docs/experiments/planning/EXPERIMENT_REGISTRY.md 后
git commit -m "metrics: Wikitext dim256 fused grid + registry"
git push origin master
```

---

## 9. 【优先】**A2-S3** 多种子 + **leaf_heldout**（**3090 fused**，与 path-batch **分列**）

与 **`PHASE1_MANUSCRIPT.md` §10** 第 1 条一致：固定 **`--cohort sibling`**（或另开 **`root_child`** 登记），**`--pair-split leaf_heldout --heldout-leaves 6`**（**16 叶** 时 test 叶对 **C(6,2)=15**；**32 叶** 更稳）。**注意**：**`leaf_heldout` 不用 ``--split-seed``**（划分由叶下标决定）；多种子请扫 **`--init-seed`**（只随机化 **reader** 权重，Wikitext 树嵌入仍由文本哈希决定）。

**前置**：**§0**（**`conda activate mamba2`**、**`MAMBA2_RESULTS_ROOT`**、**`HF_ENDPOINT`**）。若报 **`causal_conv1d` … strides … multiples of 8**：先 **`git pull`** — `Mamba2PathReader` 默认已改为 **fused 友好** 的 **num_heads≥8** 拆分（见 **`src/rag_tree/readers.py`**）。

**16 叶 × chunk8 × dim128**（示例，**5 个 init 种子**）：

```bash
cd /root/autodl-tmp/mamba2
STAMP=$(date -u +%Y%m%dT%H%MZ)
for S in 0 1 2 3 4; do
  python scripts/research/task_wikitext_path_pair.py \
    --num-leaves 16 --fanout 2 --chunk-len 8 --dim 128 \
    --cohort sibling --pair-split leaf_heldout --heldout-leaves 6 --init-seed "$S" \
    --out-json "$MAMBA2_RESULTS_ROOT/metrics/task_wikitext_sibling16_c8_leafheldout6_initseed${S}_${STAMP}.json"
done
```

**32 叶**（把 **`--num-leaves 32`**，文件名里 **`sibling32`**；**heldout 6** 仍合法）。

**若用 ``--pair-split stratified``**：再扫 **`--split-seed`** 才有意义（train/test **叶对** 划分会变）。

**跑后**：将 **`$MAMBA2_RESULTS_ROOT/metrics/task_wikitext_sibling*_initseed*.json`** 拷到本仓 **`results/metrics_result/`**（与 **Wikitext path-batch** 归档同目录），运行 **`python scripts/research/aggregate_task_wikitext_path_pair_json.py -g 'results/metrics_result/task_wikitext_sibling16_c8_leafheldout6_initseed*_YOURSTAMP.json'`**（**32** 同理）。**已归档例**：**`STAMP=20260409T1438Z`** → **10** 个 **`…_initseed{0..4}_20260409T1438Z.json`**；登记 **A-stage2-wikitext-path-pair-initseed5-3090-v1**。

---

## 10. 【辅线】SSGS × Mamba × **Wikitext 同建树**

与 **`benchmark_wikitext_tree.py`** **同一** **`wikitext2_leaf_chunks` → `build_bottom_up_text_tree`** 流程，在树上跑 **`dfs_ssgs_mamba`**（**token 步进** + **`DynamicCache`** 快照）。**测量轴**：与 **path-batch 墙钟**、**§7 单列毫秒** **分列**；JSON **`kind=ssgs_mamba_wikitext_tree`**。登记 **X-20260407-ssgs-mamba-wikitext-tree**。

**前置**：**§0**（**`HF_ENDPOINT`**、**datasets** 可拉 **Wikitext-2**）。CUDA 上 **`build_toy_mamba2_for_ssgs`** 行为与 **`demo_ssgs_mamba_dfs.py`** 一致（见 **`SERVER_SWEEP_RUNBOOK`** **SSGS** 段）。

**一键（CUDA + 汇总 CSV；可选顺带 path-batch smoke）**：

```bash
cd /root/autodl-tmp/mamba2
bash scripts/server/bootstrap_autodl.sh
RUN_WIKITEXT_SMOKE=1 bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
# 多叶数：EXTRA_LEAVES="16 32" bash scripts/server/run_ssgs_mamba_wikitext_cuda.sh
```

**单条命令（等价 n8 一格，须已 §0）**：

```bash
cd /root/autodl-tmp/mamba2
STAMP=$(date -u +%Y%m%dT%H%MZ)
python scripts/research/demo_ssgs_mamba_wikitext.py \
  --device cuda \
  --num-leaves 8 --fanout 2 --chunk-len 8 --dim 128 --layers 2 \
  --target-leaf-index -1 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics_result/ssgs_mamba_wikitext_n8_c8_${STAMP}.json"
```

本机 **CPU smoke**（小模型，快）：加 **`--cpu`**，**`--dim 64 --chunk-len 4`** 亦可。单测：**`pytest tests/test_ssgs_mamba_wikitext.py`**（**Windows** 上 **PyTorch DLL 损坏** 时 **`_mamba2_available`** 会 **skip**，不阻塞 **`pytest` 收集**）。

**多份 JSON → 一张表**（与 path-batch 的 grid CSV 并列归档；默认 **只写本次 glob 匹配的文件**；**`--append`** 则与已有 CSV **按 `json_path` 合并**）：

```bash
cd /root/autodl-tmp/mamba2
python scripts/research/aggregate_ssgs_mamba_wikitext_json.py \
  -g "$MAMBA2_RESULTS_ROOT/metrics_result/ssgs_mamba_wikitext_*.json" \
  --out-csv "$MAMBA2_RESULTS_ROOT/metrics_result/ssgs_mamba_wikitext_grid.csv"
# 若 grid 已存在、只想追加新 STAMP：
#   同上命令加 --append
```

单测（无 torch）：**`pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py`**。

**已跑通归档例**（**`git_sha` 以 JSON 为准**）：**n8** CPU+CUDA **`1529Z`**，**n16/n32** CUDA **`1550Z`**，**n64** CUDA **`1601Z`** → **`ssgs_mamba_wikitext_grid.csv`** **5 行**（登记 **X-20260407**）。可选：**n128**（大叶、耗时更长）；**`git pull` 后** 重跑 **n8** 一格对齐 **HEAD**。

### 10.1 Phase **M1**：**SSGS vs TF-KV**（**三臂**）叶数扫

与 **§10** **同 Wikitext 建树**，**`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`**：**Mamba** + **TF-KV clone** + **TF-KV truncate**（**`tf_kv_truncate_arm`** = **`truncate_kv`** 回退）。**`kind=ssgs_vs_kv_tree_nav_wikitext`**。工具表与脚注：**`SSGS_MAINLINE_M1.md`**；登记 **X-ssgs-vs-kv-tree-nav-m1**。

**前置**：**§0**。

```bash
cd /root/autodl-tmp/mamba2
bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
# 叶数：M1_LEAVES="8 16 32 64" bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
# 仅两臂（无 truncate 键）：M1_NO_TRUNCATE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
# 固定 STAMP：M1_STAMP=20260407T1200Z bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
```

产出：**`$MAMBA2_RESULTS_ROOT/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n{N}_cuda_3arm_${STAMP}.json`**（**`M1_NO_TRUNCATE=1`** 时文件名为 **`…_2arm_${STAMP}.json`**）。**跑后**：在 **`EXPERIMENT_REGISTRY`** 更新 **M1** 行或附 **n16/n32** 一句；跨臂墙钟 **不对等** 须在脚注中写明。

**汇总表**：拷入本仓 **`results/metrics_result/`** 后，**`python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py -g 'results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json' --out-csv results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv`** → **`ssgs_vs_kv_wikitext_nav_grid.csv`**（含 **L3** 列若 JSON 有 **`l3_tf_kv_hidden`**）。单测：**`pytest tests/test_aggregate_ssgs_vs_kv_wikitext_json.py`**。**`run_m1_ssgs_vs_kv_wikitext_cuda.sh`** 默认在叶扫结束后调用聚合（**`SKIP_M1_AGGREGATE=1`** 跳过；**`AGGREGATE_APPEND=1`** 追加）。

**可选 L3（隐状态一致性，非 CE）**：在 **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** 上加 **`--l3-tf-kv-hidden`**，JSON 增加 **`l3_tf_kv_hidden`**（**clone** / **truncate** 臂各 **余弦** vs 金路径-only）。见 **`SSGS_MAINLINE_M1.md`** §2.1、**`tf_kv_l3_probe.py`**。

**可选 L3 下游（固定叶头 CE）**：**`--l3-tf-kv-downstream-ce`** → **`l3_tf_kv_downstream_ce`**（未训练 **`Linear(dim,num_leaves)`**，**`abs_ce_delta`** 应极小）。与 **树 LM** **X-20260423 / X-20260424**（**CE 路由 vs 可学习子头**）**不同 harness**，登记与成文勿混表。云端：**`M1_WITH_L3_DOWNSTREAM_CE=1`**。

**L3 下游 CE · 叶数扩展（已归档例）**：**n8** **`STAMP=20260410T1113Z`**；**n16 / n32** **`STAMP=20260410T1133Z`**（**`abs_ce_delta`=0**，见 **`SSGS_MAINLINE_M1.md`** §2.1、**登记册 M1**）。若重跑或补 **n64**：

```bash
cd /root/autodl-tmp/mamba2 && git pull
M1_LEAVES="16 32" M1_WITH_L3_DOWNSTREAM_CE=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
# 例：M1_LEAVES="64" M1_WITH_L3_DOWNSTREAM_CE=1 …
```

可选同时开隐状态 L3：**`M1_WITH_L3=1 M1_WITH_L3_DOWNSTREAM_CE=1`**。**跑后**：将 JSON 拷入本仓 **`results/metrics_result/`**，本机 **`aggregate_ssgs_vs_kv_wikitext_json.py`**（勿直接提交云端 CSV 的 **`/root/...` `json_path`**），再更新 **`EXPERIMENT_REGISTRY`** / **`SSGS_MAINLINE_M1.md`**。

### 10.2 **M2** 后续实验（**M1** 已归档后的默认顺序）

**详表**：**`docs/experiments/planning/SSGS_MAINLINE_M1.md` §6**（含 **§6.0：B2 与「全链条」分层** — **必读**）。

**若要先验证「M1 全链条能跑通」再上探针**：在 AutoDL 上 **先** **`M1_LEAVES="64" bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh`**（**不开** **`M1_WITH_L3_*`**，确认三臂 **`ok`**）**再** **`M1_LEAVES="64" M1_WITH_L3_DOWNSTREAM_CE=1 bash …`**（**B2** = **环节 ②+③**；**不是**新对比方法）。**本仓已含** **`…_n64_cuda_3arm_20260410T1247Z.json`**（**`l3_tf_kv_downstream_ce`**）时，**B2** 多为 **换机复现 / 刷新 `git_sha`**；**无该文件** 时再作 **首次补档**。

**其它常跑**：**`M1_LEAVES="8"`** + 新 **`M1_STAMP`**（**`git_sha` 刷新**）。**chunk_len≠8** 的 M1：**直调** **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py --chunk-len 12`**（见 **`SSGS_MAINLINE_M1` §6** **B3**）。

---

## 11. 本机 **RTX 5060**（Windows；服务器忙时）

**全文**：**`docs/environment/runbooks/LOCAL_5060_RUNBOOK.md`**（**conda `mamba2`**、**B-S2+**、**Wikitext smoke**、**SSGS CPU**、**pytest**）。与 **§6**（B-S2+）互补：**§6** 给单行示例，**`LOCAL_5060_RUNBOOK`** 含 **WinError 1114 / base torch** 说明与 **多组 out-json 名**。

---

## 12. 阶段 5：仓内核对 + 重聚合 + 测试（**Linux / AutoDL** 推荐）

在**仓库根**执行（**`conda activate mamba2`** 或同等 **torch+cuda**）：

```bash
# 0) 可选：确认 §A2 关键文件存在（bash）
test -f results/metrics/figures/mamba_3090_naive_vs_fused_dim128_paper_main_v1.png
test -f results/metrics_result/tf_kv_trajectory_l3_minimal_cuda_20260410T1341Z.json

# 1) 将 grid CSV 的 json_path 改为仓内 POSIX 路径（覆盖写回同名 CSV）
python scripts/research/aggregate_ssgs_mamba_wikitext_json.py \
  -g 'results/metrics_result/ssgs_mamba_wikitext_*.json' \
  --out-csv results/metrics_result/ssgs_mamba_wikitext_grid.csv
python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py \
  -g 'results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json' \
  --out-csv results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv
# 期望：脚本打印 wrote ... (N row(s)) — **N = 数据行**（无表头）；**N** 随通配到的 JSON 个数变（例 **13–17**）；**json_path** 列应为 **results/metrics_result/...**

# 2) 全量单测（须 torch；AutoDL 常无 pytest 入口，勿用裸 pytest）
python -m pytest tests/ -q

# 2b) 无 torch 快测（聚合 + 几何；本机 Windows 可跑）
# py -3 -m pytest tests/test_aggregate_ssgs_mamba_wikitext_json.py \
#   tests/test_aggregate_ssgs_vs_kv_wikitext_json.py tests/test_path_pair_geometry.py -q

# 3) 可选：git_sha 刷新 — M1 单格（改 STAMP / 路径见 §10.1）
# STAMP=$(date -u +%Y%m%dT%H%MZ)
# python scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py --device cuda \
#   --num-leaves 8 --out-json results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda_3arm_${STAMP}.json
# 然后重复步骤 1) 中第二条 aggregate
```

**说明**：**§A2** 完整表与登记 id 见 **`docs/overview/execution/SUBMISSION_PACK.md` §A2**；**M1 / SSGS 数据行数** 以 **本步 `aggregate_*` stdout 的 `N row(s)`** 为准（**勿与旧稿硬编码数字打架**）。

**终端**：从聊天或 Markdown **复制多行**时，**只粘贴命令与 `#` 注释**；若把**中文说明句**（如「若还要……」）粘进 shell，会出现 **`bash: … command not found`**（该行被当成命令名）。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-09 | 初版：公共前置、HEAD 单格、**dim256** 脚本、n32 可选、B-S2+、成文指针 |
| 2026-04-09 | **§9**：**A2-S3** + **leaf_heldout**；**§0** **`metrics/`**；**多种子 = `--init-seed`**（非 **`--split-seed`**） |
| 2026-04-09 | **§9 跑后**：**`aggregate_task_wikitext_path_pair_json.py`**；登记 **A-stage2-wikitext-path-pair-initseed5-3090-v1** |
| 2026-04-09 | **§9**：拷入 **`metrics_result/`**；**`STAMP=20260409T1438Z`** 例 |
| 2026-04-09 | **已跑通**：**dim256** **`STAMP=20260409T1137Z`**；**n32** 单点；**headcheck** **`20260409T1135Z`** — 见 **`EXPERIMENT_REGISTRY`** |
| 2026-04-07 | **叶数扫描** **`run_server_wikitext_leavescale.sh`**、**§7 depth 5–6** **`run_server_section7_depth_sweep.sh`**；**`SERVER_SWEEP_RUNBOOK` §2f–§2g**；本节重编号 §4–§8 |
| 2026-04-09 | **已跑通叶数扫描**：**`TAG=stage2_leavescale`** **`STAMP=20260409T1257Z`** → **`benchmark_wikitext_stage2_leavescale_*`**；登记 **A-stage2-wikitext-leavescale-v1**；**`SERVER_SWEEP_RUNBOOK` §1** 补 **GitHub TLS / PyCharm** |
| 2026-04-09 | **§4** 末：**128/256 叶** 建议；**`run_server_wikitext_leavescale.sh`** 支持 **`CHARS_PER_LEAF`**；**`SERVER_SWEEP_RUNBOOK` §2f** 大叶数段 |
| 2026-04-09 | **已跑通 XL**：**`TAG=stage2_leavescale_xl`** **`STAMP=20260409T1322Z`（n128）**、**`20260409T1324Z`（n256）**；**`benchmark_wikitext_stage2_leavescale_xl_grid_n128_n256_combined.csv`**；**A-stage2-wikitext-leavescale-xl-v1** |
| 2026-04-07 | **§10**：**`demo_ssgs_mamba_wikitext.py`** + 登记 **X-20260407-ssgs-mamba-wikitext-tree** |
| 2026-04-07 | **§10**：**`aggregate_ssgs_mamba_wikitext_json.py`** → **`ssgs_mamba_wikitext_grid.csv`** |
| 2026-04-07 | **§10**：已跑通 **1529Z / 1550Z / 1601Z** 归档例；**n128**、**HEAD** 重跑 为可选 |
| 2026-04-07 | **§11**：**`LOCAL_5060_RUNBOOK.md`**（本机 5060 / Windows） |
| 2026-04-07 | **§0 / §10**：**`scripts/server/bootstrap_autodl.sh`**、**`run_ssgs_mamba_wikitext_cuda.sh`**（重启后环境 + 主线 SSGS CUDA） |
| 2026-04-07 | **§10.1**：**M1** **`run_m1_ssgs_vs_kv_wikitext_cuda.sh`**（**n8/n16/n32** 三臂）；**`SSGS_MAINLINE_M1.md`** |
| 2026-04-10 | **§10.1**：**`--l3-tf-kv-hidden`**（**`l3_tf_kv_hidden`**）；**`tf_kv_l3_probe.py`** |
| 2026-04-10 | **§10.1**：**`aggregate_ssgs_vs_kv_wikitext_json.py`** → **`ssgs_vs_kv_wikitext_nav_grid.csv`**；**`SKIP_M1_AGGREGATE` / `AGGREGATE_APPEND`** |
| 2026-04-10 | **§10.1**：**`--l3-tf-kv-downstream-ce`**、**`M1_WITH_L3_DOWNSTREAM_CE`**（**`l3_tf_kv_downstream_ce`**） |
| 2026-04-11 | **§10.1**：**L3 CE 叶扩** — **`STAMP=20260410T1133Z`** **n16/n32** 已归档；块内改为重跑 / **n64** 模板 |
| 2026-04-11 | **篇首公平性**：**`FIGURE_CAPTIONS`** **七轴** 指针 |
| 2026-04-11 | **§0.5**：按 **`RESEARCH_STATUS` §3.5** **L1–L4** 分层的服务器实验序（块 **A–F**：smoke、M1 **n64**、叶扫、SSGS **n128**、M1 **L3**、**B-S2+ CUDA**） |
| 2026-04-11 | **§0.5 块 F**：**`--out-json`** 改默认示例为 **`metrics_result/probe_…cuda_${STAMP}.json`**（与 **`metrics/`** 分列同步习惯对齐） |
| 2026-04-11 | **§0.5 块 G**：**阶段 C** **`benchmark_tf_kv_trajectory_l3_minimal.py`**（**L3 轨迹**） |
| 2026-04-11 | **§12**：阶段 5 **重聚合 M1/SSGS grid** + **`pytest tests/`** + 可选 M1 smoke |
| 2026-04-11 | **§10.2**：**M2** 跑道指针（**`SSGS_MAINLINE_M1` §6**；**n64+L3 CE** / **`git_sha`** / **chunk_len**） |
| 2026-04-11 | **§10.2**：**§6.0** 互链 — **先 n64 三臂 smoke 再 B2**；**B2 ≠ 全链条唯一验证** |
| 2026-04-11 | **§12**：**M1 grid** **N** 例 **13–17**；**勿将中文说明粘进 shell**；**`json_path`** 仓根 **`aggregate_*`** 重写 |
