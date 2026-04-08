# Linux（AutoDL）安装 `causal-conv1d` + `mamba-ssm`（融合核）

## 有什么好处？

| 没装（当前 HF 行为） | 装好之后 |
|----------------------|----------|
| Transformers 里 Mamba/Mamba-2 走 **naive PyTorch 回退** | 走 **CUDA 融合算子**（`selective_scan` 等） |
| **更慢**、**峰值显存往往更高**（你曾在 3090 上看到树路径 Mamba2 ~2.2GB peak） | 通常 **更快、更省显存**（具体随序列长与 batch 变） |
| 控制台反复提示 *fast path is not available* | 该提示 **消失或显著减少** |
| 仍可做 smoke / 对比实验 | 更接近 **论文与官方实现** 的训练/推理环境 |

> 说明：本仓库的 **`Mamba2PathReader`** 和 **`smoke_mamba_minimal.py`** 都通过 Transformers 调 Mamba-2；装好 `mamba-ssm` 后，同一脚本一般会**自动**用快路径（无需改 Python 代码）。

**实测（AutoDL / RTX 3090 / torch 2.11+cu126）**：装融合核后，`smoke_mamba_minimal` 峰值显存约 **56 MiB**（naive 约 **411 MiB**）；`benchmark_tree_walk --depth 4 --fanout 2` 中 Mamba2 路径 reader 峰值约 **73 MiB**（naive 约 **2248 MiB**）。具体以你机器为准。

---

## 前置条件

在 **`conda activate mamba2`** 且已装好 **GPU 版 PyTorch** 的前提下操作：

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
```

记下 **torch 版本** 与 **CUDA 后缀**（如 `cu126`）。`mamba-ssm` / `causal-conv1d` 常以 **源码编译** 形式安装，需本机有 **GCC、Python 开发头文件**；AutoDL 镜像一般已具备。

---

## 推荐安装顺序（先试 PyPI，失败再源码）

### 1. 编译常用依赖

```bash
conda activate mamba2
cd /root/autodl-tmp/mamba2
python -m pip install -U pip wheel packaging ninja
# PyTorch 2.11 官方轮常要求 setuptools<82；勿盲目 -U 到 82+，否则 pip 会报依赖冲突
python -m pip install "setuptools>=70,<82"
```

若你已误装 `setuptools 82.x`，先执行上一行再装 `causal-conv1d`。

### 2. 安装 `causal-conv1d`（先于 `mamba-ssm`）

**先试 wheel（快）：**

```bash
python -m pip install "causal-conv1d>=1.4.0" --no-build-isolation
```

若报错（版本与 torch 不匹配、无对应 wheel），再试 **从源码**（需 `git`）：

```bash
python -m pip install "git+https://github.com/Dao-AILab/causal-conv1d.git" --no-build-isolation
```

若提示找不到 CUDA，可显式指定（路径以实例为准，常见之一）：

```bash
export CUDA_HOME=/usr/local/cuda
# 或 AutoDL: ls /usr/local | grep cuda
```

### 3. 安装 `mamba-ssm`

**先试 PyPI：**

```bash
python -m pip install mamba-ssm --no-build-isolation
```

失败时到 [state-spaces/mamba Releases](https://github.com/state-spaces/mamba/releases) 看是否有 **与你的 Python 版本 + torch + CUDA** 匹配的 `.whl`，再：

```bash
python -m pip install /path/to/downloaded.whl
```

或从源码（耗时长，5–20 分钟常见）：

```bash
python -m pip install "git+https://github.com/state-spaces/mamba.git" --no-build-isolation
```

编译卡顿时可限制并行度：

```bash
export MAX_JOBS=4
```

---

## 4. 验证

```bash
python -c "import causal_conv1d; print('causal_conv1d ok')"
python -c "import mamba_ssm; print('mamba_ssm ok', getattr(mamba_ssm, '__version__', ''))"
python scripts/smoke/smoke_mamba_minimal.py --reps 3 --warmup 1
python scripts/benchmarks/benchmark_tree_walk.py --depth 4 --fanout 2 --reps 3 --warmup 1
```

对比安装前后的 **`per_step_s`** 与 **`peak_alloc_mib`（mamba2）**，并把新 CSV 命名如 `sweep_autodl_fused.csv` 以免覆盖旧结果。

---

## 5. 常见问题

- **`torch ... requires setuptools<82, but you have setuptools 82.x`**：**要管。** 执行 `python -m pip install "setuptools>=70,<82"`，与上节一致；否则少数场景下编译/运行可能出怪问题。  
- **`Building wheel for causal-conv1d ...` 卡住很久**：**正常。** 首次多半在 **本机编译 CUDA 扩展**，常见 **5–20 分钟**（视 CPU）；`nvidia-smi` 此时往往几乎空闲。可另开终端看 CPU 是否在跑 `nvcc`/`c++`。想限并行：`export MAX_JOBS=4`。  
- **`RuntimeError: CUDA extension not built`**：当前 torch 与编译时 CUDA 不一致 → 重装匹配的 torch 或换官方 release 的 wheel。  
- **编译 OOM**：`export MAX_JOBS=2`。  
- **仍显示 naive**：确认 `python -c "import mamba_ssm"` 无报错；Transformers 版本过旧时升级：`pip install -U transformers`。

---

## 6. 参考链接

- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)  
- [mamba / mamba-ssm](https://github.com/state-spaces/mamba)
