# Linux 上跑 `.sh` 报错：`/usr/bin/env: 'bash\r': No such file or directory`

## 含义（务必记住）

报错：

```text
/usr/bin/env: 'bash\r': No such file or directory
```

表示脚本**第一行 shebang 实际是** `#!/usr/bin/env bash\r`（行尾多了 **Windows CRLF** 里的 **`\\r`**）。

Linux 内核执行脚本时，会把解释器名字解析成 **`bash\\r`**（带回车），系统里没有叫这个名字的可执行文件，于是 **`env` 找不到解释器**。这与「没装 bash」无关，**几乎总是 CRLF 问题**。

同类症状：`run_path_protocol_cuda.sh: line 6: set: pipefail` 或 `invalid option name` 与路径粘在一行 — 同样是 **`set -o pipefail` 行尾带了 `\r`**，bash 把选项名认成 `pipefail\r`。修复仍用下文 **`sed -i 's/\r$//'`**。

常见来源：

- 在 **Windows** 上编辑或保存了 `.sh`，编辑器默认 CRLF；
- **PyCharm / SFTP / WinSCP** 把本机脚本**覆盖上传**到 Linux，未强制 LF；
- 从 zip/某些网盘同步来的文本被转成 CRLF。

仓库在 **`.gitattributes`** 中已设 `*.sh text eol=lf`，**`git clone` / `git pull`** 拿到的应是 **LF**。若仍用工具覆盖上传 `.sh`，上传后请再执行下面的 **`find … sed`**。

---

## 一次性修复（在仓库根目录）

```bash
cd /path/to/mamba2
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
```

然后按需：

```bash
chmod +x scripts/benchmarks/run_server_paper_main_sweep.sh \
  scripts/benchmarks/run_server_paper_main_sweep_naive.sh \
  scripts/benchmarks/run_server_sweep_aligned.sh
```

**自检**（应无 `^M`）：

```bash
sed -n '1p' scripts/benchmarks/run_server_paper_main_sweep.sh | od -c | head -1
# 行尾应为 \n，不应出现 \r
```

---

## 预防

- Windows 侧：编辑器对 `scripts/**/*.sh` 使用 **LF**；PyCharm **Deployment** 可勾选或设置换行符为 LF。
- 优先用 **Git** 更新服务器代码，少用手动覆盖 `.sh`。

---

## 另见

- 扫参流程里顺带提到：`docs/environment/SERVER_SWEEP_RUNBOOK.md`
- Git 合并与离线同步：`docs/environment/GIT_SERVER_MERGE_UNTRACKED.md`
