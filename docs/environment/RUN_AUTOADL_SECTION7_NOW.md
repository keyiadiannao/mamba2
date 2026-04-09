# AutoDL 一键：§7 深度扩展（depth 5–6，S1–S4）

> **目的**：跑 **`run_server_section7_depth_sweep.sh`**，产出登记 **`X-section7-depth-extension-v1`** 所需 JSON + manifest。  
> **与 path-batch（Wikitext 扫叶）分列**，勿无脚注混表。  
> **注意**：若遇 **`bash\r`**，必须先 **`find … sed`**（见 **`SERVER_SWEEP_RUNBOOK.md` §1 / §2f**）。

---

## 整段复制（路径按 AutoDL 默认）

```bash
cd /root/autodl-tmp/mamba2

find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
chmod +x scripts/benchmarks/run_server_section7_depth_sweep.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
mkdir -p "$MAMBA2_RESULTS_ROOT/metrics_result"

# 与 conda 一致（可选）
export PYTHON=python

# 默认 DEPTHS="5 6"；需与 20260421 归档同 depth=4 对照时用下行代替 unset STAMP 块：
# export DEPTHS="4 5 6"

unset STAMP
./scripts/benchmarks/run_server_section7_depth_sweep.sh

ls -la "$MAMBA2_RESULTS_ROOT/metrics_result/section7_depth_"* 2>/dev/null
```

---

## 跑完后

1. 记下终端里的 **`STAMP=`**（UTC）与 **`git_sha`**（`manifest` 内亦有全 SHA）。  
2. 将 **`$MAMBA2_RESULTS_ROOT/metrics_result/section7_depth_*`** 拉回本机 **`results/metrics_result/`**。  
3. 编辑 **`docs/experiments/EXPERIMENT_REGISTRY.md`**：把 **`X-section7-depth-extension-v1`** 从「跑后填」改为完整路径 + 一句 **KV/clone/restore 随 depth 趋势**。  
4. **`git add` + `commit` + `push`**。

---

## 省时（调试）

- 只跑 **depth=5**：**`export DEPTHS="5"`**  
- 降低 TF-KV 重复：**`export KV_REPS=5`**（登记时注明）
