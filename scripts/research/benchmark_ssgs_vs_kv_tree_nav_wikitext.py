#!/usr/bin/env python3
"""
**M1 harness**：与 ``demo_ssgs_mamba_wikitext`` **同一 Wikitext 建树与 DFS 任务**，默认 **三臂**：

1. **SSGS**：``dfs_ssgs_mamba`` + ``MambaNavState``（token 步进 + DynamicCache 快照/恢复）。
2. **TF-KV clone**：``dfs_tf_kv_nav`` + ``TfKvNavState``（**全层 KV 张量 clone/恢复**）。
3. **TF-KV truncate**：``dfs_tf_kv_nav`` + ``TfKvTruncateNavState``（**``truncate_kv(keep_tokens)``** 回退，对齐 §7.2 ``--branch-truncate-demo`` 语义）。

输出 JSON（``kind=ssgs_vs_kv_tree_nav_wikitext``）：``mamba_arm``、``tf_kv_clone_arm``（原 ``tf_kv_arm`` 键名保留兼容见下）、``tf_kv_truncate_arm``（可用 ``--no-tf-kv-truncate`` 跳过第三臂）。

**脚注**：两臂 TF-KV **不是** ``TransformerPathReader``；与 Mamba 的墙钟/显存 **不对等**（步长与模型不同），勿单独宣称「KV 胜 Mamba」。

服务器示例（AutoDL，仓库在 ``~/mamba2`` 或 ``/root/autodl-tmp/mamba2``）::

  export HF_ENDPOINT=https://hf-mirror.com
  export MAMBA2_RESULTS_ROOT=/root/autodl-tmp/mamba2_results
  cd /path/to/mamba2
  python scripts/research/benchmark_ssgs_vs_kv_tree_nav_wikitext.py --device cuda \\
    --out-json results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_n8_cuda.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
from src.rag_tree.ssgs import (
    MambaNavState,
    SSGSTrace,
    build_toy_mamba2_for_ssgs,
    clear_tree_snapshots,
    dfs_ssgs_mamba,
    mount_mamba_cache_meta_on_tree,
)
from src.rag_tree.tf_kv_incremental import IncrementalCausalTransformerKV
from src.rag_tree.tf_kv_l3_probe import tf_kv_hidden_consistency_nav_vs_gold_path
from src.rag_tree.tf_kv_tree_nav import (
    TfKvNavState,
    TfKvTruncateNavState,
    dfs_tf_kv_nav,
    mount_tf_kv_meta_on_tree,
    mount_tf_kv_truncate_meta_on_tree,
)
from src.rag_tree.tree import iter_root_leaf_paths


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


def _depth_edges(num_leaves: int, fanout: int) -> int:
    d = math.log(num_leaves) / math.log(fanout)
    if abs(d - round(d)) > 1e-9:
        raise ValueError(f"num_leaves={num_leaves} must be fanout**depth for fanout={fanout}")
    return int(round(d))


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _run_mamba_arm(
    root,
    leaf_goal,
    *,
    dim: int,
    layers: int,
    dev: torch.device,
) -> tuple[bool, dict[str, object]]:
    clear_tree_snapshots(root)
    model = build_toy_mamba2_for_ssgs(dim, dev, num_layers=layers)
    state = MambaNavState(model=model)
    tr = SSGSTrace()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        ok = dfs_ssgs_mamba(
            root,
            state,
            leaf_goal=leaf_goal,
            trace=tr,
            mount_snapshot=mount_mamba_cache_meta_on_tree,
        )
    _sync(dev)
    wall_s = time.perf_counter() - t0
    peak_mib = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024**2) if dev.type == "cuda" else 0.0
    )

    return ok, {
        "ok": ok,
        "wall_s": round(wall_s, 8),
        "peak_alloc_mib": round(peak_mib, 6),
        "snapshots_taken": tr.snapshots_taken,
        "rollbacks": tr.rollbacks,
        "leaf_checks": tr.leaf_checks,
        "events": list(tr.events),
        "mamba_torch_forward_only": dev.type == "cuda",
    }


def _run_tf_kv_arm(
    root,
    leaf_goal,
    *,
    dim: int,
    tf_layers: int,
    nhead: int,
    ff_mult: int,
    dev: torch.device,
) -> tuple[bool, dict[str, object]]:
    clear_tree_snapshots(root)
    model = IncrementalCausalTransformerKV(
        dim=dim,
        nhead=nhead,
        num_layers=tf_layers,
        ff_mult=ff_mult,
    ).to(dev)
    model.eval()
    state = TfKvNavState(model=model)
    tr = SSGSTrace()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        ok = dfs_tf_kv_nav(
            root,
            state,
            leaf_goal=leaf_goal,
            trace=tr,
            mount_snapshot=mount_tf_kv_meta_on_tree,
        )
    _sync(dev)
    wall_s = time.perf_counter() - t0
    peak_mib = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024**2) if dev.type == "cuda" else 0.0
    )
    kv_final = int(state.model.kv_nbytes())

    return ok, {
        "baseline": "tf_kv_full_clone_restore",
        "ok": ok,
        "wall_s": round(wall_s, 8),
        "peak_alloc_mib": round(peak_mib, 6),
        "snapshots_taken": tr.snapshots_taken,
        "rollbacks": tr.rollbacks,
        "leaf_checks": tr.leaf_checks,
        "kv_nbytes_at_end": kv_final,
        "events": list(tr.events),
    }


def _run_tf_kv_truncate_arm(
    root,
    leaf_goal,
    *,
    dim: int,
    tf_layers: int,
    nhead: int,
    ff_mult: int,
    dev: torch.device,
) -> tuple[bool, dict[str, object]]:
    clear_tree_snapshots(root)
    model = IncrementalCausalTransformerKV(
        dim=dim,
        nhead=nhead,
        num_layers=tf_layers,
        ff_mult=ff_mult,
    ).to(dev)
    model.eval()
    state = TfKvTruncateNavState(model=model)
    tr = SSGSTrace()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        ok = dfs_tf_kv_nav(
            root,
            state,
            leaf_goal=leaf_goal,
            trace=tr,
            mount_snapshot=mount_tf_kv_truncate_meta_on_tree,
        )
    _sync(dev)
    wall_s = time.perf_counter() - t0
    peak_mib = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024**2) if dev.type == "cuda" else 0.0
    )
    kv_final = int(state.model.kv_nbytes())

    return ok, {
        "baseline": "tf_kv_truncate_kv_restore",
        "ok": ok,
        "wall_s": round(wall_s, 8),
        "peak_alloc_mib": round(peak_mib, 6),
        "snapshots_taken": tr.snapshots_taken,
        "rollbacks": tr.rollbacks,
        "leaf_checks": tr.leaf_checks,
        "truncate_kv_calls": state.truncate_kv_calls,
        "kv_nbytes_at_end": kv_final,
        "events": list(tr.events),
    }


def main() -> int:
    try:
        from transformers import Mamba2Model  # noqa: F401
    except ImportError:
        print("transformers with Mamba2Model required", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2, help="Mamba hidden layers")
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--tf-nhead", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument(
        "--target-leaf-index",
        type=int,
        default=-1,
        help="left-to-right leaf order (0..n-1); -1 = rightmost",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--no-tf-kv-truncate",
        action="store_true",
        help="skip TF-KV truncate_kv arm (only Mamba + TF-KV full clone)",
    )
    p.add_argument(
        "--l3-tf-kv-hidden",
        action="store_true",
        help=(
            "optional L3: after run, compare TF-KV last-token hidden (DFS+restore) vs "
            "gold-path-only forward (same weights); adds l3_tf_kv_hidden to JSON"
        ),
    )
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    fanout = args.fanout
    n = args.num_leaves
    _depth_edges(n, fanout)

    if args.dim % args.tf_nhead != 0:
        print("dim must be divisible by --tf-nhead", file=sys.stderr)
        return 1

    try:
        leaves = wikitext2_leaf_chunks(n, args.chars_per_leaf, config=args.wikitext_config)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    want_cuda = args.device == "cuda" and not args.cpu
    dev = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")

    root = build_bottom_up_text_tree(leaves, fanout, args.chunk_len, args.dim, dev, torch.float32)
    leaves_ordered = [path[-1] for path in iter_root_leaf_paths(root)]
    if len(leaves_ordered) != n:
        raise RuntimeError(f"leaf count mismatch {len(leaves_ordered)} != {n}")
    tidx = args.target_leaf_index
    if tidx < 0:
        tidx += n
    if not (0 <= tidx < n):
        print(f"ERROR: target_leaf_index out of range: {tidx} (n={n})", file=sys.stderr)
        return 1
    target = leaves_ordered[tidx]

    def leaf_goal(node: object) -> bool:
        return node is target

    ok_m, mamba_metrics = _run_mamba_arm(root, leaf_goal, dim=args.dim, layers=args.layers, dev=dev)
    ok_kv, tf_kv_clone_metrics = _run_tf_kv_arm(
        root,
        leaf_goal,
        dim=args.dim,
        tf_layers=args.tf_layers,
        nhead=args.tf_nhead,
        ff_mult=args.ff_mult,
        dev=dev,
    )
    ok_trunc = True
    tf_kv_trunc_metrics: dict[str, object] | None = None
    if not args.no_tf_kv_truncate:
        ok_trunc, tf_kv_trunc_metrics = _run_tf_kv_truncate_arm(
            root,
            leaf_goal,
            dim=args.dim,
            tf_layers=args.tf_layers,
            nhead=args.tf_nhead,
            ff_mult=args.ff_mult,
            dev=dev,
        )

    payload: dict[str, object] = {
        "kind": "ssgs_vs_kv_tree_nav_wikitext",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "device": str(dev),
        "tree_kind": "wikitext2_bottom_up",
        "dataset": "wikitext",
        "wikitext_config": args.wikitext_config,
        "num_leaves": n,
        "fanout": fanout,
        "chars_per_leaf": args.chars_per_leaf,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "target_leaf_index": tidx,
        "mamba_layers": args.layers,
        "tf_layers": args.tf_layers,
        "tf_nhead": args.tf_nhead,
        "ff_mult": args.ff_mult,
        "mamba_arm": mamba_metrics,
        "tf_kv_clone_arm": tf_kv_clone_metrics,
        "tf_kv_arm": tf_kv_clone_metrics,
        "note": (
            "Same DFS as demo_ssgs_mamba_wikitext. Mamba = DynamicCache clone/restore (per-token HF steps). "
            "tf_kv_clone_arm / legacy tf_kv_arm = full KV tensor clone+restore at each internal node. "
            "tf_kv_truncate_arm = truncate_kv(prefix_tokens) restore (§7 branch-truncate semantics). "
            "Wall time / peak memory across arms are not apples-to-apples; do not claim KV beats Mamba from this alone. "
            "TransformerPathReader path-batch is a separate baseline."
        ),
    }
    if tf_kv_trunc_metrics is not None:
        payload["tf_kv_truncate_arm"] = tf_kv_trunc_metrics

    if args.l3_tf_kv_hidden:
        clear_tree_snapshots(root)
        l3: dict[str, object] = {
            "kind": "tf_kv_last_hidden_vs_gold_path",
            "definition": (
                "After DFS reaches target: last-token IncrementalCausalTransformerKV hidden vs "
                "fresh same-weight model forward only on root—target path (no wrong branches). "
                "cosine≈1 means restore/truncate preserves prefix state. Not a CE/LM probe."
            ),
            "clone_arm": tf_kv_hidden_consistency_nav_vs_gold_path(
                root,
                target,
                dim=args.dim,
                tf_layers=args.tf_layers,
                nhead=args.tf_nhead,
                ff_mult=args.ff_mult,
                dev=dev,
                use_truncate_restore=False,
            ),
        }
        if not args.no_tf_kv_truncate:
            clear_tree_snapshots(root)
            l3["truncate_arm"] = tf_kv_hidden_consistency_nav_vs_gold_path(
                root,
                target,
                dim=args.dim,
                tf_layers=args.tf_layers,
                nhead=args.tf_nhead,
                ff_mult=args.ff_mult,
                dev=dev,
                use_truncate_restore=True,
            )
        payload["l3_tf_kv_hidden"] = l3
        for arm_key in ("clone_arm", "truncate_arm"):
            block = l3.get(arm_key)
            if not isinstance(block, dict):
                continue
            if not block.get("dfs_ok", False):
                print(f"ERROR: l3_tf_kv_hidden.{arm_key} dfs_ok is false: {block}", file=sys.stderr)
                return 1
            if float(block.get("cosine_last_token_hidden", 0.0)) < 0.999:
                print(f"ERROR: l3_tf_kv_hidden.{arm_key} cosine < 0.999: {block}", file=sys.stderr)
                return 1

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json is not None:
        out_path = args.out_json
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print("wrote", out_path, file=sys.stderr)

    if not (ok_m and ok_kv and ok_trunc):
        print("ERROR: one or more arms failed to reach target", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
