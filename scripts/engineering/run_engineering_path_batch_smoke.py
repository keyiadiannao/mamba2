#!/usr/bin/env python3
"""
工程北星 **G1** 最小入口：封装 **`benchmark_wikitext_tree`** 单格，产出 **`kind=engineering_path_batch_smoke`** 信封 JSON。

不复制建树逻辑；**`import`** **`build_wikitext_tree_benchmark_dict`**。后续 Sprint 在同一脚本上增加 **HF Causal LM 臂** / **预训权重**（见 **`docs/overview/engineering/ENGINEERING_NORTH_STAR_PLAN.md`**）。

  python scripts/engineering/run_engineering_path_batch_smoke.py --out-json results/metrics_result/engineering/eng_path_batch_smoke.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag_tree.engineering_envelope import wrap_path_batch_smoke_envelope  # noqa: E402

_BWT_PATH = _REPO_ROOT / "scripts" / "benchmarks" / "benchmark_wikitext_tree.py"
_spec = importlib.util.spec_from_file_location("_benchmark_wikitext_tree_eng", _BWT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_BWT_PATH}")
_bwt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bwt)
build_wikitext_tree_benchmark_dict = _bwt.build_wikitext_tree_benchmark_dict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mamba2", action="store_true")
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="write engineering envelope JSON (UTF-8)",
    )
    p.add_argument("--git-sha", type=str, default=None, metavar="SHA")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    _ = args.seed

    try:
        payload = build_wikitext_tree_benchmark_dict(args)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    env = wrap_path_batch_smoke_envelope(
        payload,
        runner_relpath="scripts/engineering/run_engineering_path_batch_smoke.py",
    )
    text = json.dumps(env, indent=2)
    print(text)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
