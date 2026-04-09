#!/usr/bin/env python3
"""
A2-S3 minimal **task metric** on the **same** Wikitext-2 shallow tree as
``benchmark_wikitext_tree.py``: binary classification on **unordered leaf pairs**
whether two leaves lie in the same **fixed-height cohort** (partition block).

- **root_child**: same subtree under a child of the root
  (block size ``fanout ** (depth - 1)``).
- **sibling**: share the same parent (block size ``fanout``; only when ``depth >= 2``).

Features: **concat** of path-reader pooled vectors ``[z_i, z_j]`` (+ raw mean-pool baseline).
**Ridge** linear classifier; **not** comparable to phase-1 path-batch wall-clock tables.

Example::

  conda run -n mamba2 python scripts/research/task_wikitext_path_pair.py --cpu \\
    --num-leaves 8 --cohort sibling --out-json results/metrics/task_wikitext_path_pair_sibling8_smoke.json

**Leaf heldout** (recommended for less pair leakage): train pairs use only leaves
``0..n-h-1``, test pairs only ``n-h..n-1`` (disjoint leaf sets; same tree build).
**``--split-seed`` does not affect** this split (deterministic by leaf index). For **multiple
random trials**, vary ``--init-seed`` (reader weight init); tree embeddings stay text-deterministic.

  ... --pair-split leaf_heldout --heldout-leaves 4 --num-leaves 16 --cohort sibling

Requires: ``datasets`` (Wikitext), ``torch``, ``transformers`` (for readers).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
from src.rag_tree.path_pair_geometry import (
    all_unordered_pairs,
    block_size,
    depth_edges,
    pair_same_cohort_label,
    pairs_within_leaf_range,
)
from src.rag_tree.readers import GRUPathReader, Mamba2PathReader, TransformerPathReader, mamba2_path_reader_available
from src.rag_tree.tree import batched_paths


def _git_short_sha(repo: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return "unknown"


def _stratified_pair_split(
    pairs: list[tuple[int, int]],
    y: np.ndarray,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def _ridge_weights(X: np.ndarray, y_pm: np.ndarray, lam: float) -> np.ndarray:
    n, d = X.shape
    a = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    b = X.T @ y_pm
    return np.linalg.solve(a, b)


def _acc(X: np.ndarray, y_pm: np.ndarray, w: np.ndarray) -> float:
    pred = np.sign(X @ w)
    pred[pred == 0] = 1
    return float(np.mean(pred == y_pm))


def _make_readers(
    *,
    dim: int,
    nhead: int,
    tf_layers: int,
    gru_layers: int,
    mamba_layers: int,
    mamba_hidden: int,
    include_mamba: bool,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    readers: dict[str, torch.nn.Module] = {
        "transformer": TransformerPathReader(dim=dim, nhead=nhead, num_layers=tf_layers),
        "gru": GRUPathReader(dim=dim, num_layers=gru_layers),
    }
    if include_mamba:
        readers["mamba2"] = Mamba2PathReader(
            dim=dim,
            mamba_hidden=mamba_hidden,
            num_layers=mamba_layers,
        )
    for m in readers.values():
        m.to(device)
    return readers


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--mamba-hidden", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mamba2", action="store_true")
    p.add_argument("--cohort", choices=("root_child", "sibling", "custom"), default="sibling")
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="with --cohort custom: leaf block size for same-cohort (must divide leaf ordering)",
    )
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--ridge-lambda", type=float, default=1e-2)
    p.add_argument("--init-seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument(
        "--pair-split",
        choices=("stratified", "leaf_heldout"),
        default="stratified",
        help="stratified: random stratified split of all pairs (default). "
        "leaf_heldout: train pairs only from leaves [0,n-h), test only from [n-h,n); disjoint leaves.",
    )
    p.add_argument(
        "--heldout-leaves",
        type=int,
        default=None,
        metavar="H",
        help="with --pair-split leaf_heldout: count h of held-out leaves at the end of the index order; "
        "train uses leaves 0..n-h-1, test uses n-h..n-1. Required for leaf_heldout.",
    )
    p.add_argument("--out-json", type=Path, default=None, metavar="PATH")
    p.add_argument("--git-sha", type=str, default=None)
    args = p.parse_args()

    n = args.num_leaves
    fanout = args.fanout
    depth = depth_edges(n, fanout)

    try:
        leaves = wikitext2_leaf_chunks(n, args.chars_per_leaf, config=args.wikitext_config)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        block = block_size(args.cohort, fanout, depth, custom=args.block_size)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    root = build_bottom_up_text_tree(leaves, fanout, args.chunk_len, args.dim, device, torch.float32)
    paths, num_paths = batched_paths(root)
    if num_paths != n:
        raise RuntimeError(f"path mismatch {num_paths} != {n}")

    if args.pair_split == "leaf_heldout":
        if args.heldout_leaves is None:
            print("ERROR: --heldout-leaves H required with --pair-split leaf_heldout.", file=sys.stderr)
            return 1
        h = args.heldout_leaves
        n_tr = n - h
        if h < 2 or n_tr < 2:
            print("ERROR: need heldout_leaves>=2 and num_leaves-heldout_leaves>=2.", file=sys.stderr)
            return 1
        pairs_train = pairs_within_leaf_range(0, n_tr)
        pairs_test = pairs_within_leaf_range(n_tr, n)
        pairs = pairs_train + pairs_test
        y_train = np.array([pair_same_cohort_label(i, j, block) for i, j in pairs_train], dtype=np.int64)
        y_test = np.array([pair_same_cohort_label(i, j, block) for i, j in pairs_test], dtype=np.int64)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(
                "ERROR: leaf_heldout split has single class in train or test; "
                "try different --heldout-leaves or --cohort.",
                file=sys.stderr,
            )
            return 1
        y = np.concatenate([y_train, y_test])
        train_idx = np.arange(len(pairs_train), dtype=np.int64)
        test_idx = np.arange(len(pairs_train), len(pairs), dtype=np.int64)
        split_meta = {
            "pair_split": "leaf_heldout",
            "heldout_leaves": h,
            "train_leaf_range": [0, n_tr],
            "test_leaf_range": [n_tr, n],
            "n_train_pairs": int(len(pairs_train)),
            "n_test_pairs": int(len(pairs_test)),
            # ``--split-seed`` does not reshuffle leaf_heldout; use ``--init-seed`` for multi-seed reader inits.
            "split_seed_used_for_pair_indices": False,
        }
    else:
        pairs = all_unordered_pairs(n)
        y = np.array([pair_same_cohort_label(i, j, block) for i, j in pairs], dtype=np.int64)
        if len(np.unique(y)) < 2:
            print("ERROR: single class for all pairs; choose another --cohort or num-leaves.", file=sys.stderr)
            return 1

        train_idx, test_idx = _stratified_pair_split(pairs, y, args.test_frac, args.split_seed)
        if len(train_idx) < 4 or len(test_idx) < 2:
            print("ERROR: train/test too small.", file=sys.stderr)
            return 1
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            print("ERROR: missing class in train or test; try different --split-seed.", file=sys.stderr)
            return 1
        split_meta = {
            "pair_split": "stratified",
            "heldout_leaves": None,
            "test_frac": args.test_frac,
            "split_seed": args.split_seed,
            "split_seed_used_for_pair_indices": True,
            "n_train_pairs": int(len(train_idx)),
            "n_test_pairs": int(len(test_idx)),
        }

    y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    include_mamba = mamba2_path_reader_available() and not args.no_mamba2

    torch.manual_seed(args.init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.init_seed)

    x = paths.to(device)
    readers = _make_readers(
        dim=args.dim,
        nhead=args.nhead,
        tf_layers=args.tf_layers,
        gru_layers=args.gru_layers,
        mamba_layers=args.mamba_layers,
        mamba_hidden=args.mamba_hidden,
        include_mamba=include_mamba,
        device=device,
    )

    def pairwise_concat(z: torch.Tensor, pair_list: list[tuple[int, int]]) -> np.ndarray:
        """z: [L, D] -> features for each pair [P, 2D]."""
        zc = []
        for i, j in pair_list:
            zi = z[i].float().cpu().numpy()
            zj = z[j].float().cpu().numpy()
            zc.append(np.concatenate([zi, zj], axis=0))
        return np.stack(zc, axis=0)

    # Row order matches `pairs` (train block then test block for leaf_heldout).
    pair_list_for_matrix = pairs

    results: dict[str, object] = {}
    with torch.no_grad():
        raw_mean = x.mean(dim=1)
        results["baseline_raw_mean_concat"] = pairwise_concat(raw_mean, pair_list_for_matrix)

    with torch.no_grad():
        for name, model in readers.items():
            model.eval()
            z = model(x)
            results[name] = pairwise_concat(z, pair_list_for_matrix)

    def ridge_block(name: str, X: np.ndarray) -> dict[str, float]:
        w = _ridge_weights(X[train_idx], y_pm[train_idx], args.ridge_lambda)
        return {
            "train_acc": _acc(X[train_idx], y_pm[train_idx], w),
            "test_acc": _acc(X[test_idx], y_pm[test_idx], w),
            "feature_dim": int(X.shape[1]),
        }

    ridge_out: dict[str, object] = {k: ridge_block(k, v) for k, v in results.items()}

    sha = args.git_sha if args.git_sha is not None else _git_short_sha(_REPO_ROOT)
    payload: dict[str, object] = {
        "kind": "task_wikitext_path_pair",
        "stage": "A2-S3",
        "git_sha": sha,
        "torch_version": torch.__version__,
        "device": str(device),
        "dataset": "wikitext",
        "wikitext_config": args.wikitext_config,
        "tree_kind": "wikitext2_bottom_up",
        "task": {
            "name": "leaf_pair_same_cohort",
            "cohort": args.cohort,
            "block_size": block,
            "depth_edges": depth,
            "description": (
                "y=1 iff floor(i/block)==floor(j/block) in left-to-right leaf order "
                f"(block={block}; root_child/sibling per NEXT_RESEARCH_PLAN A2-S3)."
            ),
        },
        "num_leaves": n,
        "fanout": fanout,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "num_pairs": len(pairs),
        "pair_class_balance": {"positives": int(y.sum()), "negatives": int(len(y) - y.sum())},
        "split": split_meta,
        "n_train_pairs": split_meta["n_train_pairs"],
        "n_test_pairs": split_meta["n_test_pairs"],
        "test_frac": args.test_frac if args.pair_split == "stratified" else None,
        # Only stratified splits consume ``--split-seed``; leaf_heldout: use ``null`` (not argparse default 0).
        "split_seed": args.split_seed if args.pair_split == "stratified" else None,
        "ridge_lambda": args.ridge_lambda,
        "init_seed": args.init_seed,
        "readers_included": list(readers.keys()),
        "ridge_concat": ridge_out,
        "notes": (
            "A2-S3 path-pair task; same corpus/tree build as benchmark_wikitext_tree.py. "
            "Do not merge with phase-1 wall-clock or 3090 fused grids without labeling columns."
            + (
                " pair_split leaf_heldout: train/test pairs use disjoint leaf index sets; "
                "embeddings for all leaves still from one forward over the full tree."
                if args.pair_split == "leaf_heldout"
                else ""
            )
        ),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
