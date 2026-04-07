# 数据与样例

## 原则

- **大语料与向量索引**不入 Git；根路径由 `MAMBA2_DATA_ROOT` 指定（见 `SYNC_AND_ENVIRONMENTS.md`）。
- 仓库内仅保留 **可公开、体积小** 的样例，用于脚本联调。

## 当前样例

| 路径 | 用途 |
|------|------|
| `experiments/A-20260408-text-shaped-tree/leaves_sample.txt` | 8 条叶文本（`benchmark_text_tree.py`） |

## 后续（待填）

- 第一个「真」评测集名称、协议、下载方式与许可。
- 是否在 `MAMBA2_DATA_ROOT/raw/` 存全文，在 `processed/` 存分块与树 JSON。
