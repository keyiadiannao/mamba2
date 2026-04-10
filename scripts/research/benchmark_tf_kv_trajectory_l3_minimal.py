#!/usr/bin/env python3
"""
**阶段 C · 最小 L3**：玩具 TF-KV 上 **轨迹甲（错枝 + restore）** vs **轨迹乙（金路径直达）**。

见 **``RESEARCH_STATUS_AND_DIRECTION.md`` §3.5**、**``src/rag_tree/tf_kv_trajectory_l3.py``**。

**kind** = **``tf_kv_trajectory_l3_minimal``**；与 **path-batch**、**M1** **分列**登记。

Examples::

  python scripts/research/benchmark_tf_kv_trajectory_l3_minimal.py --device cpu --out-json results/metrics_result/tf_kv_trajectory_l3_minimal_cpu.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.rag_tree.tf_kv_trajectory_l3 import compare_trajectories_ab  # noqa: E402


def _git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--depth", type=int, default=2, help="balanced tree depth (2 => 4 leaves for fanout 2)")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--tf-nhead", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--init-seed", type=int, default=42)
    p.add_argument("--out-json", type=str, required=True)
    args = p.parse_args()

    dev = torch.device(args.device)

    clone_payload, _ = compare_trajectories_ab(
        depth=args.depth,
        fanout=args.fanout,
        chunk_len=args.chunk_len,
        dim=args.dim,
        tf_layers=args.tf_layers,
        nhead=args.tf_nhead,
        ff_mult=args.ff_mult,
        device=dev,
        init_seed=args.init_seed,
        use_truncate_restore=False,
    )
    trunc_payload, _ = compare_trajectories_ab(
        depth=args.depth,
        fanout=args.fanout,
        chunk_len=args.chunk_len,
        dim=args.dim,
        tf_layers=args.tf_layers,
        nhead=args.tf_nhead,
        ff_mult=args.ff_mult,
        device=dev,
        init_seed=args.init_seed,
        use_truncate_restore=True,
    )

    out = {
        "kind": "tf_kv_trajectory_l3_minimal",
        "git_sha": _git_short_sha(_REPO),
        "torch_version": torch.__version__,
        "device": str(dev),
        "tree": {
            "depth": args.depth,
            "fanout": args.fanout,
            "chunk_len": args.chunk_len,
            "dim": args.dim,
            "target_leaf": "last_in_iter_root_leaf_paths",
        },
        "tf_trunk": {
            "layers": args.tf_layers,
            "nhead": args.tf_nhead,
            "ff_mult": args.ff_mult,
        },
        "init_seed": args.init_seed,
        "notes": (
            "Trajectory A takes wrong sibling at root (one node), then restore, then gold suffix. "
            "Not full M1 DFS; explicit L3 control-flow probe per RESEARCH_STATUS §3.5."
        ),
        "clone_restore_arm": clone_payload,
        "truncate_kv_arm": trunc_payload,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
