# 服务器上 `merge` / `pull` 失败

文件名仍用 `GIT_SERVER_MERGE_UNTRACKED`；下面分两种常见报错。

---

## A. 「Please commit your changes or stash them before you merge」

典型完整信息里还会有 **Your local changes to the following files would be overwritten by merge** 和一串已跟踪文件路径。

含义：这些文件在服务器上 **已被修改或已暂存**，`pull` 要合并入站版本时会覆盖它们，Git 直接中止。**与 Windows 本地无关**，只看你 **服务器** 这个 clone 是否干净。

### 先看状态（在仓库根目录）

```bash
cd /path/to/mamba2
git status
```

若显示 `You have unmerged paths` 或正在进行 merge，先中止再处理：

```bash
git merge --abort 2>/dev/null || true
```

### 方案 1：不要服务器上的改动，只要和 GitHub 一致（最常见）

**会丢弃工作区里所有未提交修改和未跟踪文件**（先备份 `results/metrics` 等）：

```bash
git fetch origin
git reset --hard origin/master
git clean -fd
```

然后再：

```bash
git pull origin master
```

（`reset --hard` 后通常已与 `origin/master` 一致，`pull` 可能只是 “Already up to date”。）

### 方案 2：想暂时保留服务器上的修改

```bash
git stash push -u -m "server wip before pull"
git pull origin master
git stash pop   # 若有冲突再手动解决
```

`-u` 会把未跟踪文件也收进 stash，避免和入站文件再次冲突。

---

## B. 「untracked … would be overwritten by merge」

若出现：

```text
error: The following untracked working tree files would be overwritten by merge:
        docs/README.md
        ...
Please move or remove them before you merge.
```

说明当前目录里这些路径是 **未跟踪副本**（例如手工解压、半套 `git clone`、或曾把文件拷进空仓库），与 **远端已有跟踪文件** 同名，Git 无法自动覆盖。

---

## 推荐做法（服务器上只要跟 GitHub 一致）

在仓库根目录执行：

```bash
cd ~/autodl-tmp/mamba2   # 按你的路径改

# 1) 若 results/metrics 里有仅存在于本机的数据，先备份
mkdir -p ~/autodl-tmp/mamba2_metrics_backup
cp -r results/metrics/* ~/autodl-tmp/mamba2_metrics_backup/ 2>/dev/null || true

# 2) 删掉所有未跟踪文件/目录（不删已提交历史）
git clean -fd

# 3) 再拉取
git fetch origin
git pull origin master
```

若仍异常，可强制与远端对齐（**会丢掉未 push 的本地提交**）：

```bash
git fetch origin
git reset --hard origin/master
git clean -fd
```

---

## 更干净：重新克隆（数据盘）

```bash
cd /root/autodl-tmp
mv mamba2 mamba2_old_$(date +%Y%m%d%H%M)   # 旧目录改名备份
git clone https://github.com/keyiadiannao/mamba2.git
cd mamba2
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mamba2
# 再按 AUTODL_SETUP / SERVER_SWEEP_RUNBOOK 装依赖
```

---

## 拉取成功后

```bash
chmod +x scripts/benchmarks/run_server_paper_main_sweep.sh
# 若报 bash\r：
# sed -i 's/\r$//' scripts/benchmarks/run_server_paper_main_sweep.sh
WARMUP=2 REPS=8 TAG=paper_main_v1 ./scripts/benchmarks/run_server_paper_main_sweep.sh
```

---

## `stash` 何时有用

- **A 类**（已跟踪文件有修改）：`git stash -u` 再 `pull` 合适。
- **B 类**（纯未跟踪挡路）：优先 **`git clean -fd`** 或移走目录；若你坚持用 stash，需要 **`git stash push -u`** 把未跟踪也收起来，否则仍会冲突。

---

## C. 服务器访问 GitHub 超时（`Failed to connect to github.com port 443`）

说明机器到 GitHub 的 HTTPS 不通（防火墙/地区线路/机房策略）。**不必在服务器上 `git pull`**，可以用 **本机已同步的仓库** 覆盖到服务器。

### 做法 1：PyCharm Deployment（SFTP/SSH）

1. 在 **Windows** 上先把本地 `mamba2` **`git pull` 到最新**。
2. PyCharm：**Settings → Build, Execution, Deployment → Deployment**，新增 **SFTP**，填服务器 SSH 主机、用户、认证方式，**Root path** 指到服务器上的项目目录（如 `/root/autodl-tmp/mamba2`）。
3. **Mappings**：Local path = 本机仓库根目录，Deployment path = `/`（相对 Root path）或等价映射。
4. 右键项目根目录 → **Deployment → Upload to…**（或 **Sync with Deployed to…** 先看差异再上传）。
5. **排除**（在 Deployment → Excluded Paths）：`.git` 可选（见下）、`.venv` / `venv`、`__pycache__`、`.idea`、大体积 `data/` 等，避免慢传或覆盖环境。

**是否上传 `.git`：**

- **上传 `.git`**：服务器上的 Git 历史与本机一致，以后若网络恢复可继续 `git pull`（注意本机要先提交/拉齐再传）。
- **不上传 `.git`**：只保证 **代码与脚本** 与本地一致；服务器上 `git status` 可能无意义，以「能跑 benchmark」为主即可。

上传后可在 SSH 里执行：

```bash
chmod +x scripts/benchmarks/*.sh
sed -i 's/\r$//' scripts/benchmarks/*.sh   # 若出现 bash\r
```

### 做法 2：本机打包 + SCP / WinSCP

在本机仓库上级目录：

```bash
# PowerShell 示例：排除虚拟环境（按你目录名调整）
tar -czvf mamba2-sync.tgz --exclude=.venv --exclude=__pycache__ -C d:\cursor_try mamba2
```

用 **WinSCP** 或 `scp` 把 `mamba2-sync.tgz` 传到服务器后解压到目标目录（或解压到新目录再替换）。

### 做法 3：Git (bundle) 离线包（本机能连 GitHub 时在本机做）

在本机：

```bash
git bundle create mamba2-master.bundle origin/master
```

把 `mamba2-master.bundle` 拷到服务器后：

```bash
git fetch ../mamba2-master.bundle master:refs/remotes/origin/master
git reset --hard origin/master
```

（路径按你放置 bundle 的位置调整。）
