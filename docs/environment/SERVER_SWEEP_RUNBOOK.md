# 服务器空闲后：对齐扫参（fused）操作手册

> 目标：在 **Linux + `mamba_ssm` 融合栈** 上复跑与本地 **HF naive** 相同或可对键合并的网格，生成 **naive vs fused** 主文级对照数据。  
> 前置安装见 **`MAMBA_SSM_INSTALL_LINUX.md`**。

---

## 1. 更新代码

```bash
cd /path/to/mamba2   # 或 git clone 见 AUTODL_SETUP.md
git pull origin master
source /root/miniconda3/etc/profile.d/conda.sh   # 避免「Run conda init before conda activate」
conda activate mamba2
```

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

若报错 **`/usr/bin/env: 'bash\r': No such file or directory`**，说明脚本为 Windows CRLF 行尾，在服务器上执行：

```bash
sed -i 's/\r$//' scripts/benchmarks/run_server_sweep_aligned.sh
```

然后重新 `chmod +x` 并运行。仓库已加 **`.gitattributes`**（`*.sh` 强制 LF）；请 **`git pull`** 后再跑。

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

（将路径换成你下载后的实际文件名。）

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

## 5. 仅跑子集（省时）

若只想快速验证 fused，只跑 A：

```bash
python scripts/benchmarks/sweep_tree_benchmark.py --preset local --dim 128 --warmup 2 --reps 8 \
  --out-csv results/metrics/sweep_adl_dim128_quick.csv
```
