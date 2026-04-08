# 双机环境（5060 8GB × AutoDL 48GB）与同步规范

## 1. 分工原则

| 位置 | 角色 | 适合做什么 |
|------|------|------------|
| **本地 5060 8G** | 代码编辑、Git、小数据调试、单步调试、文档 | 极小 batch、短序列、mock 树、单元测试 |
| **AutoDL 48G** | 正式训练、长上下文、大批量、树构建（若吃显存） | 主实验、checkpoint 产出 |

**原则**：能在本地验证「逻辑正确」的，不要上云；上云只跑已本地 smoke 通过的提交（commit hash 写入 registry）。

**本机项目环境（Windows）**：conda 环境 **`mamba2`**，解释器 `C:\Users\26433\miniconda3\envs\mamba2\python.exe`。创建与 PyTorch 版本（**5060 需 `+cu128`**）见 `environment/MAMBA2.md`。仓库根目录执行 `python scripts/smoke/smoke_local.py` 做冒烟。

---

## 2. 什么走 Git，什么不走

| 走 Git | 不走 Git（各自机器或网盘） |
|--------|------------------------------|
| `src/`, `configs/`, `scripts/`, `docs/`, `experiments/*/README.md` 与小配置 | `data/raw`, `data/processed`, `checkpoints/`, 大 `results/` |
| `environment/requirements.txt` + lock | 权重文件、完整语料 |

---

## 3. 推荐路径约定（减少换机改代码）

- 在**每台机器**上通过**环境变量**指向数据与输出根目录，代码里只读 `os.environ` 或 Hydra 默认：
  - `MAMBA2_DATA_ROOT`：语料与中间文件根
  - `MAMBA2_CKPT_ROOT`：checkpoint 根
  - `MAMBA2_RESULTS_ROOT`：指标与日志根
- `configs/` 里使用相对占位或 `${oc.env:MAMBA2_DATA_ROOT}`（若后续采用 OmegaConf）。

**本地示例（PowerShell）**

```powershell
$env:MAMBA2_DATA_ROOT = "D:\cursor_try\mamba2_data"
$env:MAMBA2_CKPT_ROOT = "D:\cursor_try\mamba2_ckpt"
$env:MAMBA2_RESULTS_ROOT = "D:\cursor_try\mamba2_results"
```

**AutoDL 示例（bash）**：将上述路径改为数据盘挂载点，例如 `/root/autodl-tmp/mamba2_data`（以你实例为准）。

---

## 4. 同步方式（按优先级）

1. **代码**：Git（GitHub / Gitee / 自建）。AutoDL 上 `git pull`；本地 `git push`。大文件勿提交。
2. **大文件（数据、ckpt）**：
   - AutoDL 面板「公网网盘」或与机器同区域的对象存储；**同一数据集只传一次**，在 registry 记录版本与 MD5（可选）。
   - 本地 8G **不要**全量同步 checkpoint；只拉取评测所需**单个**小 ckpt 时用手动下载。
3. **结果回传**：优先只同步 `results/metrics/*.json`、`experiments/*/README.md`、小图；日志可用压缩包。

**Windows 侧**：可选 WinSCP、FileZilla、`rclone`；与脚本模板见 `scripts/sync/sync_example.ps1`。

**Linux 侧**：`rsync -avz --progress` 或 `rclone copy`；模板见 `scripts/sync/sync_example.sh`。

---

## 5. 上传下载纪律（避免乱）

- 每次上传数据/模型前：在 `docs/experiments/EXPERIMENT_REGISTRY.md` 记 **文件名 + 目标路径 + 用途**。
- 训练产出目录命名：`${MAMBA2_CKPT_ROOT}/run_{YYYYMMDD}_{exp_id}/`。
- 回传本地时只取**登记过的 run**，避免磁盘被占满。

---

## 6. 显存策略速查

| 场景 | 5060 8G | AutoDL 48G |
|------|---------|------------|
| 调试 forward | 最小序列、gradient checkpointing、fp16/bf16 | 正常开发 batch |
| 训练检索头/适配器 | 仅适配器可试；全参微调优先云上 | 主训 |

---

## 7. AutoDL 快速上手

分步命令见 **`docs/environment/AUTODL_SETUP.md`**（克隆、`conda`、PyTorch、`smoke`、扫参、CSV 回传与合并）。

---

## 8. 待填实例信息（填后勿提交含密码文件）

- AutoDL 数据盘挂载路径：___________
- 网盘/对象存储 bucket（如有）：___________
- 默认 Git 远程：`https://github.com/keyiadiannao/mamba2.git`（分支 `master`）
