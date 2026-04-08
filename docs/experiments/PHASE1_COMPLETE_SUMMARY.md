# 阶段 1 完整总结（存档）

> **用途**：论文/报告「阶段 1 做了什么」的一页式叙事 + 复现入口；与 **`PHASE1_VALIDATION_PLAN.md`**（扫参细则）、**`FIGURE_CAPTIONS_STAGE1.md`**（图注句稿）、**`RESEARCH_NOTES.md` §7**（协议数学）分工不同 — 本文件不重写公式，只收束**结论与路径**。

---

## 1. 阶段 1 在答什么问题

在 **树状索引 + 固定 reader 协议** 下，比较 **Transformer / GRU / Mamba-2** 作为 **路径编码器** 时，**延迟**与 **CUDA 峰值显存** 随 **叶数 batch、chunk、深度、宽度** 的变化；并证明 **Mamba-2 的可观测效率对实现路径（HF naive vs `mamba_ssm` fused）极度敏感**。  

**不在阶段 1 内完结**：全 LLM KV 分项、生产级检索头训练、真 LM 导航的「强基线」声明 — 后者已另册登记 **X-20260422–25**（辅线）。

---

## 2. 已完成交付物（与登记册对应）

| 类别 | 内容 | 登记 / 路径 |
|------|------|-------------|
| **主文 path-batch** | 3090 同机 **naive vs fused** 扫参 + 三张主图 | **A-20260408-paper-main-3090-{fused,naive,pair}**；CSV 见 **`PHASE1_VALIDATION_PLAN.md` §6.5**；图 `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png` |
| **语料浅树** | Wikitext-2 叶块 + **同一 reader harness** | **A-20260408-wikitext-3090-fused** |
| **§7 玩具协议** | S1–S4 单路径、depth4/chunk8/dim128，3090 CUDA JSON | **X-20260421-***；仓内归档 `results/metrics/*_20260421.json` |
| **SSGS 导航环** | `dfs_ssgs_mamba` + `DynamicCache` 快照/恢复 | **X-20260421-ssgs-mamba-dfs-demo** 等 |
| **叙事护栏** | 主图 vs §7 vs SSGS vs 真 LM 四线边界 | **`FIGURE_CAPTIONS_STAGE1.md`** 篇首、`RESEARCH_NOTES` §7.0 |

**结论文本（可进正文）**：见 **`PHASE1_VALIDATION_PLAN.md` §6.3** — 要点：**naive Mamba path reader 可达 GiB 级峰值；fused 同网格可降至约 10²MiB 量级；效率结论须写「实现敏感」**。

---

## 3. 口径边界（写论文时必读）

1. **path-batch 主图**：度量的是 **沿根—叶路径批量前向** 的 harness，**不是** DFS 试错序，**不是** §7 表里某一列的毫秒物理意义。  
2. **§7.3.1 表**：S1/S2/S3/S4 **各列不同**，**禁止**与主图曲线混为同一「一步」或互相相减得出结论。  
3. **5060 与 3090** CSV **不可**混填为同一张主表的格点；跨机图仅可作 **动机/趋势**。  
4. **真 LM 线（X-20260422–25）**：辅线；**reach_rate&lt;1** 的默认设定**不得**宣称「已解决树上导航」。

---

## 4. 遗留与下一阶段

- **§7 串行复跑**：见下文 **附录 A**（新 GPU 上建议跑一遍，生成带时间戳 JSON，与 `*_20260421.json` 对照）。  
- **阶段 2**：真数据浅层树 + **同一 reader 槽位** + **至少一个任务指标** — 见 **`docs/overview/ROADMAP.md`「阶段 2 入口（一页）」**。

---

## 附录 A：§7 玩具协议 CUDA 串行复跑（完整指令）

**环境**：**Linux**（AutoDL / 3090 等），已装 **NVIDIA 驱动 + CUDA**；仓库已 `git clone`；**bash** 可用。脚本路径：`scripts/research/run_path_protocol_cuda.sh`。

**若在 Windows 本机**：请 **不要** 用 PowerShell 直接跑 `.sh`；应在 **WSL2** 或 **上传到 AutoDL** 后执行。若从 Windows 克隆的脚本报 **`$'\r': command not found`**，见 **`docs/environment/SH_CRLF_LINUX.md`**（换行符需 LF）。

### A.1 推荐：AutoDL / 服务器一次性执行

```bash
# 0) 进入仓库（路径按你机器修改）
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2
cd /root/autodl-tmp/mamba2

# 1) 可选：Hub 镜像（本脚本不拉模型权重则通常不必；若子环境 import 触发 hub 可设）
export HF_ENDPOINT=https://hf-mirror.com

# 2) 可选：输出写到数据盘（否则默认写入仓库内 results/metrics/）
# export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results

# 3) 可选：指定 Python（默认命令 python）
# export PYTHON=python3

# 4) 串行复跑 S1→S4（需 GPU；约数分钟量级，视机器而定）
bash scripts/research/run_path_protocol_cuda.sh
```

**成功标志**：终端打印 `Done. Compare with archived...`，且在 **`$MAMBA2_RESULTS_ROOT/metrics`**（若已设置）或 **`results/metrics/`** 下出现若干带 **UTC 时间戳** 的文件，例如：

- `mamba2_cache_snap_segments_depth4_cuda_<STAMP>.json`（S1）  
- `tf_r1_path_segments_depth4_cuda_<STAMP>.json`（S2）  
- `tf_kv_path_segments_depth4_cuda_<STAMP>.json`（S3 线性路径）  
- `tf_kv_path_segments_depth4_cuda_branchdemo_<STAMP>.json`（S3 错枝截断演示）  
- `mamba2_cache_restore_depth4_cuda_same_<STAMP>.json`（S4 GPU 快照）  
- `mamba2_cache_restore_depth4_cuda_fromcpu_<STAMP>.json`（S4 CPU 快照）

**对照**：与仓内归档 **`results/metrics/mamba2_cache_snap_segments_depth4_cuda_20260421.json`** 等同名协议文件、以及 **`RESEARCH_NOTES.md` §7.3.1** 表中数量级对比。

### A.2 依赖与故障排除

- **依赖**：`transformers`（含 **Mamba2** 相关）、**PyTorch CUDA**；S1/S4 走 HF **`Mamba2Model` + `DynamicCache`**。若使用 **fused 栈**，需环境与 **`EXPERIMENT_REGISTRY`** 中 **3090 fused** 行一致。  
- **`torch.cuda.is_available()` 为 False**：脚本仍会调用 `--device cuda` 而失败 — 请在 **GPU 实例**上执行。  
- **OOM**：先减小 `--depth` 或 `--dim`（需改 `.sh` 内参数，并**另开登记行**说明与默认 **X-20260421** 网格不一致）。

- **`set: pipefail` / `invalid option name` 与文件名粘在一起**：几乎总是 **Windows CRLF** 检出 — 行尾 `\r` 使 `pipefail` 变成非法选项。在仓库根执行：
  ```bash
  sed -i 's/\r$//' scripts/research/run_path_protocol_cuda.sh
  ```
  然后重跑 `bash scripts/research/run_path_protocol_cuda.sh`。长期做法见 **`docs/environment/SH_CRLF_LINUX.md`**；仓内 **`.gitattributes`** 已设 `*.sh text eol=lf`，**`git pull` 最新脚本**后应不再出现。

### A.3 与阶段 1 叙事的关系

本附录产出 **仅服务 §7 玩具协议复现**，**不替代** **A-20260408-paper-main-3090-*** 主图数据；正文引用时须 **分列标注**（见上文第 3 节）。

---

## 附录 B：§7 复跑验收（与 `RESEARCH_NOTES` §7.3.1 对照）

**场景**：AutoDL CUDA、`MAMBA2_RESULTS_ROOT` 指向数据盘时，脚本将带 **`STAMP`** 的 JSON 写入 **`$MAMBA2_RESULTS_ROOT/metrics/`**（与仓内 `results/metrics/*_20260421.json` **并存**，不必覆盖归档）。

**验收**：下列量级应与 **§7.3.1** 及 **`results/metrics/*_20260421.json`** **同阶**（允许 ±30% 级波动，因 GPU 负载与驱动差异）：

| 协议 | 抽查字段（seg 0→4） | 你次复跑（示例） | §7.3.1 表（约） |
|------|---------------------|------------------|-----------------|
| S1 | `clone_wall_ms` | ~0.14→0.08 ms | 0.16→0.075 |
| S2 | `forward_mean_ms` | ~0.53–0.56 ms | 0.54–0.58 |
| S3 | `increment_last_chunk_mean_ms` | ~2.0→9.9 ms | 2.0→10.1 |
| S4 same | `restore_wall_ms` | ~0.051 ms | 0.051–0.068 |
| S4 CPU | `restore_wall_ms` | ~0.13 ms | 0.13–0.17 |
| S3 demo | `truncate_mean_ms` | ~0.10 ms | ~0.11 |

**结论**：若上表一致，则 **§7 玩具协议在当前 fused 环境下可复现**；**`git_sha`** 在 JSON 内可为历史提交（如 **`6fa7873`**），与当前 **HEAD** 不同 **不**影响与登记 **X-20260421-*** 的口径对齐，但正文应写清 **复现所用 commit 或归档文件名**。
