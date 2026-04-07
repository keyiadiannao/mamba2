# Mamba × 树状 RAG × 检索头 — 研究仓库

本地开发与小规模验证（RTX 5060 8GB），训练与主实验在 AutoDL（48GB）。**代码与配置以 Git 为唯一真相源；大文件走网盘/面板同步，不进仓库。**

- 全局目标、分类与阶段计划：`docs/PROJECT_OVERVIEW.md`
- 双机分工与上传下载规范：`docs/SYNC_AND_ENVIRONMENTS.md`
- 实验登记（随做随记）：`docs/EXPERIMENT_REGISTRY.md`

## 本机 Python（Miniconda）

```powershell
C:\Users\26433\miniconda3\python.exe scripts\smoke_local.py
```

若 `torch` 为 `+cpu` 而本机有 NVIDIA 显卡，请到 [pytorch.org](https://pytorch.org/get-started/locally/) 按 CUDA 版本安装 **GPU 版** PyTorch，再重跑 smoke。
