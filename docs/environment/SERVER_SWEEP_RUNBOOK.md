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
