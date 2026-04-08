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
