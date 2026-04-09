# 服务器空闲后：对齐扫参（fused）操作手册

> 目标：在 **Linux + `mamba_ssm` 融合栈** 上复跑与本地 **HF naive** 相同或可对键合并的网格，生成 **naive vs fused** 主文级对照数据。  
> 前置安装见 **`MAMBA_SSM_INSTALL_LINUX.md`**。

若执行任何 `.sh` 时出现 **`/usr/bin/env: 'bash\r': No such file or directory`**：脚本为 **Windows CRLF**，Linux 把解释器当成 `bash\r`。**专文与修复命令**见 **`SH_CRLF_LINUX.md`**（勿与「未安装 bash」混淆）。

---

## 1. 更新代码

```bash
cd /path/to/mamba2   # 或 git clone 见 AUTODL_SETUP.md
git pull origin master
source /root/miniconda3/etc/profile.d/conda.sh   # 避免「Run conda init before conda activate」
conda activate mamba2
```

**若 `git pull` 失败**（例：**`GnuTLS recv error (-110)`**、TLS 被机房掐断）：在 **本机** **`git pull`** 后，用 **PyCharm SFTP / 部署** 把仓库同步到服务器即可。此时服务器上 **`git rev-parse HEAD`** 可能 **仍显示旧提交**（**`.git` 未更新**），属预期；**benchmark JSON** 里的 **`git_sha`** 会与 **`rev-parse --short`** 一致、**但不一定等于 GitHub 最新主线**。**登记**时以 **本次跑出的 manifest 全 SHA**、**结果文件路径** 与 **你在正文写的代码版本说明** 为准；需要 **`HF_TOKEN`** 时可降 Hub 匿名限速告警（见 **`HF_ENDPOINT`** 与 Hub 文档）。

确认融合路径生效（**不应**再刷「fast path is not available」一整段；或 `python -c "import mamba_ssm"` 无错）：

```bash
python scripts/smoke/smoke_mamba_minimal.py --reps 2 --warmup 1
```

---

## 2. 一键扫参（推荐）

在仓库**根目录**执行：

```bash
chmod +x scripts/benchmarks/run_server_sweep_aligned.sh
TAG=autodl_fused_$(date +%Y%m%d) ./scripts/benchmarks/run_server_sweep_aligned.sh
```

若报错 **`/usr/bin/env: 'bash\r': No such file or directory`**，说明脚本为 Windows CRLF 行尾（常见于 **PyCharm/SFTP 上传**）。在仓库根目录一次性去掉所有 `scripts/**/*.sh` 的 `\r`：

```bash
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_sweep_aligned.sh scripts/benchmarks/run_server_paper_main_sweep.sh scripts/benchmarks/run_server_paper_main_sweep_naive.sh
```

然后重新运行。仓库 **`.gitattributes`** 已设 `*.sh text eol=lf`；**`git pull` / clone** 应得到 LF。若仍从 Windows **覆盖上传** `.sh`，再跑上述 `find … sed` 即可。

会依次写出 **4 份 CSV**（文件名含 `$TAG`）：

| 段 | 内容 | 对齐本地文件（示例） |
|----|------|----------------------|
| A | `preset local`，`dim=128`，8 点 | `sweep_tree_reader_20260410_local5060.csv`（同键） |
| B | `dim=256`，chunk 4/8/12，12 点 | `sweep_local5060_dim256_chunk_sweep_20260410.csv` |
| C | `dim=256`，chunk 含 16，`reps=8`，16 点 | `sweep_local5060_dim256_chunk4to16_reps8_20260410.csv` |
| D | `dim=384`，6 点 | `sweep_local5060_dim384_20260410.csv` |

可选：结果写到数据盘：

```bash
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics"
TAG=adl1 ./scripts/benchmarks/run_server_sweep_aligned.sh
```

### 2b. 同机主文扫参（统一 `WARMUP` / `REPS`，审稿友好）

用于论文 **Figure 1 候选**：整段会话内 **固定** `warmup` 与 `reps`，产出 `paper_main_*_${TAG}.csv` 与 **`paper_main_manifest_${TAG}.txt`**（含 `git_sha`、torch、是否加载 `mamba_ssm`）。

```bash
git pull origin master
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
chmod +x scripts/benchmarks/run_server_paper_main_sweep.sh
# 可选：sed -i 's/\r$//' scripts/benchmarks/run_server_paper_main_sweep.sh

export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results   # 可选
WARMUP=2 REPS=8 TAG=paper_main_v1 ./scripts/benchmarks/run_server_paper_main_sweep.sh
```

生成三份 CSV：**dim256×12 点**、**dim128 preset local 8 点**、**dim384×6 点**（网格与本地 extended 对齐，但 **reps 与本地历史 CSV 可能不同**，以本脚本为准写主文）。

### 2c. 同机 HF naive（替代跨机 5060 vs 3090）

`run_server_paper_main_sweep.sh` 在 **已装 `mamba_ssm` + `causal_conv1d`** 时跑的是 **融合路径**。要与 **同一 GPU、同一网格** 下的 **HF naive** 对比，需在 **另一个 conda 环境** 中卸掉融合栈后跑 **`run_server_paper_main_sweep_naive.sh`**（网格与 `WARMUP`/`REPS` 与 fused 脚本共用 `_paper_main_sweep_body.sh`，完全一致）。

**建环境（示例，在已装好 `mamba2` 的机器上）：**

```bash
conda create -n mamba2_naive --clone mamba2
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2_naive
pip uninstall -y mamba-ssm causal-conv1d
python -c "import importlib.util as u; assert u.find_spec('mamba_ssm') is None and u.find_spec('causal_conv1d') is None"
```

**跑 naive 主文扫参（建议 `WARMUP`/`REPS` 与 fused 次相同；`TAG` 用不同前缀区分）：**

```bash
cd /path/to/mamba2
git pull origin master   # 若网络可用；否则本机同步
chmod +x scripts/benchmarks/run_server_paper_main_sweep_naive.sh
sed -i 's/\r$//' scripts/benchmarks/run_server_paper_main_sweep_naive.sh  # 若遇 bash\r
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
WARMUP=2 REPS=8 TAG=paper_main_naive_v1 ./scripts/benchmarks/run_server_paper_main_sweep_naive.sh
```

**本机叠画（同机对照，主文优先）：**

```powershell
python scripts\benchmarks\plot_mamba_naive_vs_fused.py `
  --csv-a results\metrics\paper_main_naive_dim128_localgrid_paper_main_naive_v1.csv `
  --label-a "3090 HF naive same grid" `
  --csv-b results\metrics\paper_main_dim128_localgrid_paper_main_v1.csv `
  --label-b "3090 fused same grid" `
  --out results\metrics\figures\mamba_naive_vs_fused_dim128_3090_same_machine.png
```

（将路径换成你下载后的实际文件名，例如本机 `results\metrics_result\...`。）

**conda 分工（避免误卸主环境）**：`pip uninstall mamba-ssm causal-conv1d` **只应在 `mamba2_naive` 里做**。主环境 **`mamba2`** 保留融合栈，用于 `run_server_paper_main_sweep.sh`、`run_server_sweep_aligned.sh` 等。naive 跑完后若要再跑 fused：`conda activate mamba2` 即可；若曾在 **`mamba2` 主环境**误卸融合包，需按 **`MAMBA_SSM_INSTALL_LINUX.md`** 装回。

### 2d. 阶段 2 **A2-S2**：Wikitext 浅树 **fused** 小网格（与 5060 四格同拓扑）

**目的**：在 **真语料浅树**、`dim=128`、`fanout=2` 下，对 **`(num_leaves, chunk_len) ∈ {8,16}×{8,12}`**（默认 **四格**）跑 **TF / GRU / Mamba2 path reader** 的 **path-batch** 计时与 **`m2_peak`**，环境为 **3090 + fused `mamba_ssm`**。与 **5060 HF naive** 的 **`benchmark_wikitext_5060_cuda_*.json`** **分列对照**，**禁止**无脚注同表（见 **`PHASE2_DRAFT.md` §3**、**`FIGURE_CAPTIONS_STAGE1.md`** 五轴表）。

**一键（仓库根目录）**：

```bash
cd /path/to/mamba2
git pull origin master
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'   # 若遇 bash\r
chmod +x scripts/benchmarks/run_server_stage2_wikitext_grid.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
export HF_ENDPOINT=https://hf-mirror.com    # Hub 不可达时，见 AUTODL_SETUP §2b
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results   # 可选；不设则写入仓库内 results/
./scripts/benchmarks/run_server_stage2_wikitext_grid.sh
```

**常用环境变量**（脚本头注释有全文）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `WARMUP` | `2` | 与 **paper_main** 一致 |
| `REPS` | `8` | 与 **paper_main** 同量级；若要与 **5060 四格**完全同 repetitions，设 **`REPS=5`** |
| `TAG` | `stage2_fused` | 输出文件名前缀；多轮跑区分用 |
| `STAMP` | `date -u +%Y%m%dT%H%MZ` | 单次会话内四格共用同一时间戳 |
| `GRID` | `full` | **`minimal`** = 只跑 3 格（8×8、16×8、16×12），省时间 |

**产出**（在 **`$MAMBA2_RESULTS_ROOT/metrics_result`** 或 **`results/metrics_result`**）：

- 每格一份 JSON：`benchmark_wikitext_<TAG>_<STAMP>_n{leaves}_c{chunk}.json`
- 汇总 CSV：`benchmark_wikitext_<TAG>_grid_<STAMP>.csv`（脚本末尾自动调用 **`aggregate_wikitext_tree_json_grid.py`**）
- Manifest：`benchmark_wikitext_<TAG>_manifest_<STAMP>.txt`

**跑完后**：把 **`TAG` / `STAMP` / `git_sha` / GPU 型号 / 上述路径** 写回 **`EXPERIMENT_REGISTRY.md`** 行 **`A-stage2-wikitext-grid-v1`**（多轮可 **追加一段「R2」** 或 **新开登记 id**，避免覆盖首轮 **`20260409T1035Z`** 叙述）。

### 2e. **A2-S2 第二轮复跑**（对齐当前 **HEAD** / 新 **`git_sha`**）

**动机**：首轮 JSON 内 **`git_sha`** 可能落后于仓库 **HEAD**；复跑时用 **`git pull`** + 显式 **`TAG`**（如 **`stage2_fused_r2`**），便于与代码版本 **一一对应**。

**在服务器上整段复制执行**（路径按 AutoDL 默认；若你的仓库不在 **`/root/autodl-tmp/mamba2`**，只改第一处 **`cd`**）：

```bash
# 1) 代码与脚本
cd /root/autodl-tmp/mamba2
git fetch origin
git checkout master
git pull origin master

# 2) CRLF（仅在上传/Windows 覆盖过 .sh 时需要）
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_stage2_wikitext_grid.sh

# 3) Conda + fused 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

# 4) Hub（机房常用镜像）与结果目录
export HF_ENDPOINT=https://hf-mirror.com
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result"

# 可选：降低匿名限速告警、提高稳定性（有 token 时）
# export HF_TOKEN=hf_xxxxxxxx

# 5) 记录本轮提交（跑完登记册要写这行）
git rev-parse --short HEAD

# 6) 第二轮：固定 TAG，STAMP 由脚本按 UTC 自动生成
export TAG=stage2_fused_r2
export WARMUP=2
export REPS=8
unset STAMP
./scripts/benchmarks/run_server_stage2_wikitext_grid.sh

# 7) 确认产出（将下列路径与 manifest 中的 STAMP 记下来）
ls -la "$MAMBA2_RESULTS_ROOT/metrics_result/benchmark_wikitext_${TAG}"_* 2>/dev/null || \
ls -la /root/autodl-tmp/mamba2/results/metrics_result/benchmark_wikitext_${TAG}_* 2>/dev/null
```

**跑完后登记**（本机编辑 **`EXPERIMENT_REGISTRY.md`**）：在 **`A-stage2-wikitext-grid-v1`** 中 **追加**「**R2**：**`TAG=stage2_fused_r2`**、**`STAMP=…`**、**`git_sha=…`**、四 JSON + **`benchmark_wikitext_stage2_fused_r2_grid_<STAMP>.csv`**」；把 **`metrics_result/`** 下新文件 **拉回仓库** 后 **`git add` + commit**。

**本机仅重汇总 CSV**（JSON 已下载到仓库）：

```powershell
cd d:\cursor_try\mamba2
python scripts\benchmarks\aggregate_wikitext_tree_json_grid.py `
  --glob "results/metrics_result/benchmark_wikitext_stage2_fused_*_n*_c*.json" `
  --out-csv results\metrics_result\benchmark_wikitext_stage2_fused_grid_manual.csv
```

（将 **`--glob`** 换成你实际的 **`TAG`/`STAMP`** 模式。若 JSON 在数据盘 **`MAMBA2_RESULTS_ROOT\results\metrics_result`**，再加 **`--base-dir`** 指向 **`MAMBA2_RESULTS_ROOT\results`**，且 **`--glob`** 用相对形式 **`metrics_result\...`**。）

### 2f. **叶数扫描**（固定 **chunk_len**、**dim**，**path-batch**）

**目的**：在 **真语料 Wikitext 浅树**、**fused** 环境下，固定 **`chunk_len`**（默认 **8**）与 **`dim`**（默认 **128**），扫描 **`num_leaves ∈ {8,16,32,64}`**（**fanout=2**），观察 **`m2_peak` / `per_step_s`** 随 **叶数与路径长度** 的变化。与 **四格网格**（**§2d**）**分列登记**（建议 id **`A-stage2-wikitext-leavescale-v1`**）。

**叙事要点**：**`TransformerPathReader`** 对每条路径做 **整段 self-attention**（**O(T²)**）；随深度增加 **TF** 往往比 **GRU / Mamba2** 涨得更陡。详见 **`src/rag_tree/readers.py`**、**`benchmark_wikitext_tree.py`** 模块说明。

**一键**（仓库根目录）：

若出现 **`/usr/bin/env: 'bash\r': No such file or directory`**，说明 `.sh` 为 **Windows CRLF**（**PyCharm 上传** 覆盖后常见）。在仓库根 **每次上传后** 执行一次 **`find … sed`** 再去跑脚本：

```bash
cd /path/to/mamba2
git pull origin master   # 若不可用可跳过；见 §1
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_wikitext_leavescale.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
export HF_ENDPOINT=https://hf-mirror.com
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result"

export TAG=stage2_leavescale
unset STAMP
./scripts/benchmarks/run_server_wikitext_leavescale.sh
```

**可选环境变量**：

| 变量 | 默认 | 说明 |
|------|------|------|
| `LEAVES` | `8 16 32 64` | 空格分隔；**64 叶** 较慢、**TF** 更重；显存紧张时可 **`LEAVES="8 16 32"`** |
| `CHUNK_LEN` | `8` | 与阶段 2 主网格一致 |
| `DIM` | `128` | 与 **A2-S2** 主表一致；勿与 **§2f** 的 **dim256 四格**无说明混表 |
| `WARMUP` / `REPS` | `2` / `8` | 与 **paper_main** 习惯一致 |
| `CHARS_PER_LEAF` | `600` | **128/256 叶** 若语料不够可降低（与 **`benchmark_wikitext_tree.py --chars-per-leaf`** 一致） |

**产出**：**`benchmark_wikitext_<TAG>_<STAMP>_n{8,16,32,64}_c<CHUNK_LEN>.json`** + **`…_grid_<STAMP>.csv`** + manifest。

**大叶数可选（128 / 256 叶）**：

- **合法性**：**`fanout=2`** 时 **128=2⁷**、**256=2⁸**，脚本 **`LEAVES`** 可任意空格分隔幂次。
- **语料**：默认 **`chars_per_leaf=600`** → **128 叶** 需 **76800** 字符、**256 叶** 需 **153600**；一般 **Wikitext-2 raw** 够。若 **`RuntimeError: wikitext slice too short`**，设 **`CHARS_PER_LEAF=400`**（或更小）再跑（脚本已支持环境变量 **`CHARS_PER_LEAF`**）。
- **怎么做更合适**：**先 128 单点**（观察 **`m2_peak` / 是否 OOM / 墙钟**），**再 256**；不要与已归档 **8–64** 混在同一 **`TAG`**（建议 **`TAG=stage2_leavescale_xl`**），登记册 **另开一行**（如 **`A-stage2-wikitext-leavescale-xl-v1`**）。**256 叶** 上 **TF 整段 SA** + **大 batch_paths** 压力最大，若不稳可先 **`REPS=4`** 或 **`nvidia-smi -l 1`** 盯显存。
- **示例**（**128** 单会话）：

```bash
export TAG=stage2_leavescale_xl
export LEAVES="128"
unset STAMP
./scripts/benchmarks/run_server_wikitext_leavescale.sh
```

**256**（**128 绿** 后再跑）：**`export LEAVES="256"`**，其余同上；或 **`LEAVES="128 256"`** 同 **STAMP** 一次产出 **2 JSON + 1 CSV**。

### 2g. §7 协议 **S1–S4** 按 **树深度** 扩展（**单路径**，与 path-batch **分列**）

**一键粘贴（AutoDL 默认路径）**：专文 **`RUN_AUTOADL_SECTION7_NOW.md`**（含 **`sed` 去 `\r`**、**`conda`**、**`MAMBA2_RESULTS_ROOT`**、跑后登记步骤）。

**目的**：在已归档 **`depth=4`**（**`X-20260421-*`**，`RESEARCH_NOTES` / **§7**）之外，对 **`depth ∈ {5,6}`**（路径 **6 / 7** 个节点；**fanout=2** 对应 **32 / 64 叶**）各跑 **S1 Mamba clone**、**S2 TF-R1 重算**、**S3 TF-KV**、**S4 Mamba restore**，便于看 **KV 字节 / 增量步耗时 / restore** 随深度趋势。与 **`benchmark_wikitext_tree`** **不同 harness**，**禁止**与 **§2d/§2f** 无脚注同表。

**脚本说明**：**`run_server_section7_depth_sweep.sh`** 已将 JSON **标准输出丢弃**（只写文件）；**进度** 打在 **stderr**（**`-> S1 …`** 等）。可用环境变量 **`PYTHON`**（默认 **`python`**）指向 conda 解释器。

```bash
cd /path/to/mamba2
git pull origin master
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_section7_depth_sweep.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result"

unset STAMP
./scripts/benchmarks/run_server_section7_depth_sweep.sh
```

**可选**：**`DEPTHS="4 5 6"`** 与仓内 **20260421** JSON 同参再跑一遍（**STAMP** 新）；**`KV_REPS` / `RESTORE_REPS`** 与 Python 脚本默认一致（**20**）。

**产出**（**`metrics_result/`**）：**`section7_depth_s{1,2,3,4}_*_d<depth>_<STAMP>.json`** + **`section7_depth_manifest_<STAMP>.txt`**。登记建议 id **`X-section7-depth-extension-v1`**。

---

## 3. 拉回本机合并 / 出图

将 `metrics/sweep_adl_*.csv` 下载到 Windows 工程 `results/metrics/`，例如与本地 fused 旧表并列后：

**同键合并（多文件）：**

```powershell
cd d:\cursor_try\mamba2
python scripts\benchmarks\merge_sweep_csv.py results\metrics\merged_adl_local_mamba.csv results\metrics\sweep_local5060_dim256_chunk_sweep_20260410.csv results\metrics\sweep_adl_dim256_chunk412_YOURTAG.csv
```

**Mamba2 峰值对比图（与本地 naive 叠画）：**

```powershell
python scripts\benchmarks\plot_mamba_naive_vs_fused.py `
  --csv-a results\metrics\sweep_local5060_dim256_chunk_sweep_20260410.csv `
  --label-a "5060 HF naive dim256" `
  --csv-b results\metrics\sweep_adl_dim256_chunk412_YOURTAG.csv `
  --label-b "AutoDL fused dim256" `
  --out results\metrics\figures\mamba_naive_vs_fused_dim256.png
```

---

## 4. 登记

在 **`docs/experiments/EXPERIMENT_REGISTRY.md`** 新增一行：`commit`、`TAG`、GPU 名、`mamba_ssm` 版本（如有）、四份 CSV 路径、与 §6 叙事一致的一句话结论。

---

## 5. Wikitext 基准与 Hub 网络

`scripts/benchmarks/benchmark_wikitext_tree.py` 会经 `datasets` 访问 Hub。若报 **`Network is unreachable`** / 无法连 `huggingface.co`，见 **`AUTODL_SETUP.md` §2b**（`HF_ENDPOINT` / `MAMBA2_USE_HF_MIRROR=1`）。

---

## 6. 仅跑子集（省时）

若只想快速验证 fused，只跑 A：

```bash
python scripts/benchmarks/sweep_tree_benchmark.py --preset local --dim 128 --warmup 2 --reps 8 \
  --out-csv results/metrics/sweep_adl_dim128_quick.csv
```

---

## 7. 研究下一步（大叶数 + Mamba2 输出探针）

### 7a. 大叶数扫参（与 `paper_main` 区分）

`paper_main` 网格 **`max_leaves=64`**（depth≤6）。研究向 **128 条并行路径**（depth=7）扩展时用：

```bash
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'   # 若 PyCharm 上传过 .sh
chmod +x scripts/benchmarks/run_server_research_large_leaves.sh

export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
export HF_ENDPOINT=https://hf-mirror.com   # 若需镜像
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2

WARMUP=2 REPS=5 TAG=research_lg_v1 ./scripts/benchmarks/run_server_research_large_leaves.sh
```

产出：`sweep_research_large_leaves_dim128_${TAG}.csv`、`dim256_...` 与 **`sweep_research_large_leaves_manifest_${TAG}.txt`**。下回本机后写入 **`EXPERIMENT_REGISTRY`**（新 id，注明 TAG 与 fused）。

### 7b. §7.5 S1：探针 `Mamba2Model` 前向里有什么

在云端 fused 环境跑（也可本机对照）：

```bash
python scripts/research/probe_mamba2_outputs.py --device cuda
python scripts/research/probe_mamba2_outputs.py --device cuda --use-cache
python scripts/research/probe_mamba2_outputs.py --device cuda --use-cache --dump-cache
```

用于确认 **`last_hidden_state`**、**`cache_params` / DynamicCache** 等字段，再决定快照从何处 `clone`（见 `RESEARCH_NOTES` §7.1）。**不是**正式基准，无需登记；日志可贴进实验笔记或 PR 描述。

若报 **`causal_conv1d ... strides ... multiples of 8`**：多为 **fused 核在小 batch / 少 head** 上的限制。探针脚本已对 **CUDA+fused** 自动把 **`batch` 提到 ≥8**，并把 **`num_heads` 优先设为 8**（`head_dim=32`）；仍失败时可试 **`--batch 16`** 或 **`--seq 32`**（8 的倍数）。

**S1 分段快照基准**（单路径；每读完路径上第 *k* 个节点，对**累积** `inputs_embeds` `[1, (k+1)*chunk_len, dim]` 做一次完整 forward，`batch=1` 的 cache；与 path-reader 全路径大 batch 不同）：

实现上**不在段之间传入** `cache_params`：HF Mamba2 在「续写 cache」的 fused 路径里要求 **`seq_len == 1`**；若下一段仍喂整段 chunk，会报 **`causal_conv1d_update`：`weight must have shape (dim, width)`**。累积整段前向与逐步因果消费在最终 cache 上等价。

```bash
python scripts/research/benchmark_mamba2_cache_snapshot_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics/mamba2_cache_snap_segments_$(date -u +%Y%m%dT%H%MZ).json"
```

若 fused 在 **`batch=1`** 上仍触发 stride 报错，可先 **`--device cpu`** 或在正文声明该数为 CPU 协议参考；再迭代多路径 batch 化。

**S2 / §7.2 TF-R1（与 S1 同树、累积前缀）**：`TransformerPathReader` 全序列前向，**无 KV**，测每边界 `forward_mean_ms` 与 CUDA `peak_alloc_mib`。

```bash
python scripts/research/benchmark_tf_r1_path_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics/tf_r1_path_segments_depth4_cuda_$(date -u +%Y%m%dT%H%MZ).json"
```

**S3 / §7.2 TF-KV**（手写 Pre-LN 因果 trunk + 层间 KV cache；与 ``TransformerPathReader`` 非逐层同一实现，宽度/深度一致）：

```bash
python scripts/research/benchmark_tf_kv_path_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics/tf_kv_path_segments_depth4_cuda_$(date -u +%Y%m%dT%H%MZ).json"
# 可选：根下错子 → 截断 KV → 兄弟 chunk（§7.2 语义演示）
python scripts/research/benchmark_tf_kv_path_segments.py --device cuda --branch-truncate-demo \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics/tf_kv_path_segments_depth4_cuda_branchdemo_$(date -u +%Y%m%dT%H%MZ).json"
```

**SSGS × Mamba（DFS 演示，CPU/CUDA）**：

CUDA 上 ``build_toy_mamba2_for_ssgs`` 会将 mixer **固定为 ``torch_forward``**（batch=1 token 步进否则会触发 fused ``causal_conv1d`` 的 stride=8 报错）。

```bash
python scripts/research/demo_ssgs_mamba_dfs.py --device cpu
python scripts/research/demo_ssgs_mamba_dfs.py --device cuda
```

**一键复跑 S1–S4（CUDA）**（时间戳文件名写入 ``metrics/``；未设 ``MAMBA2_RESULTS_ROOT`` 时用仓库内 ``results/metrics/``）：

```bash
bash scripts/research/run_path_protocol_cuda.sh
```

**S4 / §7.3 SSM restore**（与 S1 同模型与路径；快照 ``clone`` 后对活动 ``DynamicCache`` 做 ``zero_`` → ``copy_`` 还原；**不含**重算前向）：

```bash
# 快照与模型同显存（设备内 copy_）
python scripts/research/benchmark_mamba2_cache_restore_segments.py --device cuda \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics/mamba2_cache_restore_depth4_cuda_same_$(date -u +%Y%m%dT%H%MZ).json"
# 快照在 CPU；restore 含拷回 GPU（§7.3「到设备」）
python scripts/research/benchmark_mamba2_cache_restore_segments.py --device cuda --snapshot-device cpu \
  --depth 4 --chunk-len 8 --dim 128 \
  --out-json "$MAMBA2_RESULTS_ROOT/metrics/mamba2_cache_restore_depth4_cuda_fromcpu_$(date -u +%Y%m%dT%H%MZ).json"
```
