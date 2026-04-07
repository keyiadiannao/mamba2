# 数据与样例

## 原则

- **大语料与向量索引**不入 Git；根路径由 `MAMBA2_DATA_ROOT` 指定（见 `SYNC_AND_ENVIRONMENTS.md`）。
- 仓库内仅保留 **可公开、体积极小** 的样例，用于脚本联调。

## 仓库内路径

| 路径 | 内容 | 许可 / 说明 |
|------|------|-------------|
| `data/raw/sample/*.txt` | 8 段英文合成说明文本（每文件一段） | **合成联调用**，非评测集；可自由修改 |
| `data/raw/sample/README.md` | 目录说明 | — |
| `experiments/A-20260408-text-shaped-tree/leaves_sample.txt` | 8 行叶文本（中英混合） | 合成；与上表二选一或混用 |

从 `data/raw/sample` 生成叶文件（供 `benchmark_text_tree.py`）：

```bash
python scripts/prepare_leaves_from_corpus.py \
  --input-dir data/raw/sample \
  --out results/metrics/leaves_from_repo_sample.txt \
  --fanout 2 --depth 3
```

## AutoDL / 本机大数据

- 全文、PDF、向量库放在 `$MAMBA2_DATA_ROOT/raw/` 与 `processed/`，**不提交 Git**。
- 云端首次配置步骤见 **`docs/AUTODL_SETUP.md`**。

## 后续（正式实验）

- 选定公开数据集名称、下载方式、协议与引用格式。
- 在 `EXPERIMENT_REGISTRY.md` 登记每份外源数据的版本与哈希（可选）。
