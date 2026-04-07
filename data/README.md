大体积数据不进入 Git。请在各机器设置 `MAMBA2_DATA_ROOT`，将 `raw/`、`processed/` 放在该根目录下。

**例外**：仓库内保留 `data/raw/sample/` 极小合成样例（见 `.gitignore` 白名单），仅用于联调。

推荐结构：

```
$MAMBA2_DATA_ROOT/
  raw/           # 原始语料
  processed/     # 分块、embedding、树索引等
```
