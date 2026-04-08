# 服务器上 `merge` / `pull`：untracked 将覆盖

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

## 不要用 `stash` 处理上述列表

`git stash` 默认不解决「未跟踪与入站跟踪冲突」的典型情况；**`git clean -fd`** 或 **移走冲突路径** 才是正解。
