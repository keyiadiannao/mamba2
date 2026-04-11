#!/usr/bin/env python3
"""
**工程北星统一 CLI**（**§Ⅷ-1 G1** 形式闭合）：子命令 **转发** 到既有 Runner，**不**复制实现。

| 子命令 | 脚本 |
|--------|------|
| **path-batch-smoke** | **`run_engineering_path_batch_smoke.py`** · **`kind=engineering_path_batch_smoke`** |
| **g3-compare** | **`run_g3_causal_lm_compare.py`** · **`kind=engineering_causal_lm_compare`** |
| **causal-kv-smoke** | **`run_causal_lm_path_kv_smoke.py`** · **`kind=engineering_causal_lm_path_kv_smoke`** |
| **m1-ssgs-vs-kv** | **`benchmark_ssgs_vs_kv_tree_nav_wikitext.py`** · **`kind=ssgs_vs_kv_tree_nav_wikitext`** |

用法（与直接调用子脚本等价）::

  python scripts/engineering/run_engineering.py path-batch-smoke --out-json results/metrics_result/engineering/eng.json
  python scripts/engineering/run_engineering.py g3-compare --help

详见 **`docs/overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md`**、**`scripts/engineering/README.md`**。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SUBCOMMANDS: dict[str, Path] = {
    "path-batch-smoke": _REPO_ROOT / "scripts/engineering/run_engineering_path_batch_smoke.py",
    "g3-compare": _REPO_ROOT / "scripts/engineering/run_g3_causal_lm_compare.py",
    "causal-kv-smoke": _REPO_ROOT / "scripts/engineering/run_causal_lm_path_kv_smoke.py",
    "m1-ssgs-vs-kv": _REPO_ROOT
    / "scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py",
}


def _usage() -> str:
    lines = [
        "usage: run_engineering.py <command> [args ...]",
        "",
        "Commands:",
    ]
    for name in sorted(_SUBCOMMANDS):
        lines.append(f"  {name}")
    lines.append("")
    lines.append("Run `python scripts/engineering/run_engineering.py <command> --help` for each tool.")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if len(sys.argv) >= 2 else 0

    cmd = sys.argv[1]
    if cmd not in _SUBCOMMANDS:
        print(f"run_engineering.py: unknown command {cmd!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    script = _SUBCOMMANDS[cmd]
    if not script.is_file():
        print(f"run_engineering.py: missing script {script}", file=sys.stderr)
        return 2

    rest = sys.argv[2:]
    proc = subprocess.run(
        [sys.executable, str(script), *rest],
        cwd=str(_REPO_ROOT),
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
