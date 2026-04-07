数据不进入 Git。请在各机器设置 `MAMBA2_DATA_ROOT`，将 `raw/`、`processed/` 放在该根目录下。

推荐结构：

```
$MAMBA2_DATA_ROOT/
  raw/           # 原始语料
  processed/     # 分块、embedding、树索引等
```
