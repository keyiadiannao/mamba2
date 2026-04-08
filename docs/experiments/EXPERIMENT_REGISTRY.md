# 实验登记册

> 每条实验一行（或一个小节）。**必填**：id、日期、机器、git commit、目的、关键命令、主要指标、一句话结论。

---

## 登记表

| id | 日期 | 机器 | commit | 方向 | 目的 | 关键指标 | 结论 |
|----|------|------|--------|------|------|----------|------|
| env-001 | | 5060+ADL | | X | 环境复现与 smoke | import OK | |
| X-20260407-smoke-local | 2026-04-07 | 5060 | | X | conda env mamba2 冒烟 | torch 2.11.0+cu128, CUDA OK, 50x fwd ~0.27s GPU; mamba_ssm 未装 | OK |
| A-20260407-toy-tree-reader-bench | 2026-04-07 | 5060 | | A | 玩具树三 Reader 微基准 | `scripts/benchmarks/benchmark_tree_walk.py` | TF+GRU+Mamba2PathReader（默认） |
| A-20260407-sweep-local | 2026-04-07 | 5060 | | A | 扫参 preset=local | `scripts/benchmarks/sweep_tree_benchmark.py` | 见 `results/metrics/sweep_tree_reader_20260407_local.csv` |
| A-20260408-text-shaped-tree | 2026-04-07 | 5060 | | A | 文本形树 reader 基准 | `scripts/benchmarks/benchmark_text_tree.py` | 确定性文本嵌入；待换神经 encoder |
| X-20260408-corpus-sample | 2026-04-07 | 5060 | | X | `data/raw/sample` + prepare_leaves | `scripts/data/prepare_leaves_from_corpus.py` → `scripts/benchmarks/benchmark_text_tree.py` | 合成 8 段；叶文件见 .gitignore 生成物 |
| X-20260408-autodl-doc | 2026-04-07 | — | | X | AutoDL 上手指南 | `docs/environment/AUTODL_SETUP.md` | 流程已验证；见下行云端扫参 |
| A-20260409-sweep-autodl-3090 | 2026-04-09 | AutoDL / RTX 3090 47G | `ab982d7` | A | 云端 preset=local 扫参 | `sweep_autodl.csv` 于 `results/metrics/` | torch 2.11.0+cu126；smoke OK；树基准 depth4 时 Mamba2 naive peak≈2248MiB；`mamba_ssm` 未装 |
| X-20260409-autodl-fused-mamba | 2026-04-09 | AutoDL / RTX 3090 | —（仅环境） | X | `causal_conv1d`+`mamba_ssm` 后复测 | `scripts/smoke/smoke_mamba_minimal` + `scripts/benchmarks/benchmark_tree_walk` d4 f2 | smoke peak **56MiB**（naive≈411）；树 Mamba2 peak **73MiB**（naive≈2248）；快路径生效 |
| X-20260409-mamba-minimal-smoke | 2026-04-07 | 5060 | | X | HF Mamba2Model tiny smoke（默认） | `scripts/smoke/smoke_mamba_minimal.py` | 无 mamba-ssm；`--arch mamba` 为 v1 |
| A-20260410-wikitext-shallow-tree | 2026-04-10 | 5060 | `3218cf9`+ | A | Wikitext-2 叶块 → 浅树 → 同 harness | `scripts/benchmarks/benchmark_wikitext_tree.py` | 见 `benchmark_wikitext_p0_20260410.json`；开加速器后 Hub 无 SSL 报错（仍可走缓存） |
| A-20260410-sweep-local5060 | 2026-04-10 | 5060 | `3218cf9` | A | P0 玩具树扫参 preset=local | `sweep_tree_benchmark.py --preset local --warmup 2 --reps 5` | CSV 同上；图 `results/metrics/figures/sweep_readers_20260410_local5060.png`（`plot_tree_reader_sweep.py`）；Mamba2 naive 随叶数升（d6c8 m2_peak≈8.9GiB） |
| A-20260410-sweep-local5060-ext | 2026-04-10 | 5060 | `6e84de1` | A | ~15min 窗口内多网格（HF naive） | `sweep_tree_benchmark.py` `--dim 256/384`，多 `chunk_len` / `reps` | 见 `sweep_local5060_dim256_chunk_sweep_20260410.csv`、`..._chunk4to16_reps8...`、`..._highreps...`、`..._dim384...`；d6c12 dim256 **m2_peak≈8.8GiB** |
| X-20260410-ssgs-dryrun | 2026-04-10 | 5060 | — | X | P1 SSGS + §7 协议 | `ssgs.py`：`dfs_ssgs` / `dfs_ssgs_tensor` / `TensorNavState`；`tests.test_ssgs`；`benchmark_ssgs_tensor_overhead.py` | 无 LM；张量快照=clone+restore 微基准见脚本 JSON 输出 |
| A-20260410-sweep-autodl-fused-aligned | 2026-04-08 | AutoDL / RTX 3090 | CSV 内 `ab982d7` | A | `run_server_sweep_aligned.sh` 对齐本地网格 | `sweep_adl_dim128_localgrid_autodl_fused_20260410.csv` 等 4 份 | fused **m2_peak** 相对 5060 naive 同键大降；对比图 `figures/mamba_naive_vs_fused_dim{128,256,384}*.png`（跨机，主文宜同机复扫） |
| A-20260408-paper-main-3090-fused | 2026-04-08 | AutoDL / RTX 3090 | `6fa7873` | A | 主文网格 fused：`run_server_paper_main_sweep.sh` | `TAG=paper_main_v1`，`WARMUP=2` `REPS=8`；`results/metrics_result/paper_main_{dim256,dim128_localgrid,dim384}_paper_main_v1.csv` + `paper_main_manifest_paper_main_v1.txt` | manifest：`mode=fused_expected`，`mamba_ssm` True；dim128 d6c8 **m2_peak** ≈216MiB |
| A-20260408-paper-main-3090-naive | 2026-04-08 | AutoDL / RTX 3090 | `6fa7873` | A | 同网格 HF naive：`mamba2_naive` + `run_server_paper_main_sweep_naive.sh` | `TAG=paper_main_naive_v1`，路径同上后缀 `_paper_main_naive_v1` + manifest | manifest：`mode=hf_naive`，`mamba_ssm`/`causal_conv1d` False；dim128 d6c8 **m2_peak** ≈8.9GiB |
| A-20260408-paper-main-3090-pair | 2026-04-08 | AutoDL / RTX 3090 | `6fa7873` | A | **同机** naive vs fused 主文对照（与上行两行同一轮数据） | `plot_mamba_naive_vs_fused.py` → `results/metrics/figures/mamba_3090_naive_vs_fused_dim{128,256,384}_paper_main_v1.png`（各 8/12/6 个重合格点） | 主文硬对照：同 GPU、同 commit、同计时设定；实现路径致 **m2_peak** 量级差（例 dim128 最重格 ~GiB vs ~10²MiB） |

---

## 字段说明

- **id**：与 `experiments/` 目录名后缀一致，如 `A-20260407-baseline`。
- **commit**：该次实验所用代码提交；若脏工作区，注明 `dirty` 并简述差异。
- **关键命令**：可粘贴完整一行，或指向 `experiments/.../README.md`。

---

## 待跑实验队列（ backlog ）

| 优先级 | id | 说明 |
|--------|-----|------|
| P0 | — | （已完成）同机 paper_main：见登记表 **A-20260408-paper-main-3090-*** |
| P1 | | |
