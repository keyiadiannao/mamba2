#!/usr/bin/env python3
"""
**B-S2 延伸（本地）**：在 **path-batch 同 harness** 上，对 **path reader 输出向量** 做 **岭二分类**
（与 ``probe_retrieval_correlation.py`` 的 GPT-2 探针 **并列、不可混读**）。

默认 **16 叶**、**叶模板 heldout**（每类留出若干 **未见过的叶句** 到 test），避免句级 i.i.d. 虚高。
可选 **少量训练** reader + 线性头（同一数据、仅 train 索引）再报 **BCE 分类准确率**，与 **未训练 ridge** 对照（性价比：步数默认小）。

  conda run -n mamba2 python scripts/research/probe_path_reader_linear.py --cpu --n-leaves 16 --leaf-split heldout --out-json results/metrics/probe_path_reader_linear_text16_heldout_cpu.json
  conda run -n mamba2 python scripts/research/probe_path_reader_linear.py --cpu --n-leaves 16 --leaf-split heldout --train-steps 50 --train-lr 3e-3 --out-json results/metrics/probe_path_reader_linear_text16_heldout_train50_cpu.json
  # 8 叶对照：--n-leaves 8；句级随机划分：--leaf-split sample

依赖：``torch``、``transformers``（Mamba2 reader）、``numpy``。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.readers import GRUPathReader, Mamba2PathReader, TransformerPathReader, mamba2_path_reader_available
from src.rag_tree.tree import batched_paths, iter_root_leaf_paths


def _git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, d = X.shape
    a = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    b = X.T @ y
    return np.linalg.solve(a, b)


def _accuracy_from_w(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    pred = np.sign(X @ w)
    pred[pred == 0] = 1
    return float(np.mean(pred == y))


def _stratified_indices(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
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


def _stem_and_life_pools() -> tuple[list[str], list[str]]:
    """8+8 unique lines per half; first half STEM, second half life/arts (fixed order)."""
    stem = [
        "Gradient descent minimizes loss over model parameters.",
        "Attention maps queries keys and values across token positions.",
        "Layer normalization rescales activations per token vector.",
        "KV caches avoid recomputing keys for past decoder steps.",
        "Dropout zeros random activations during training to limit co-adaptation.",
        "Mixed precision keeps master weights in float32 while tensors use half.",
        "The optimizer state tracks first and second moments per parameter.",
        "对比学习用正负样本对拉近或推远在嵌入空间中的距离。",
    ]
    life = [
        "The bread rose overnight under a linen cloth in a cool kitchen.",
        "Rowers feather their oars as the shell glides toward dawn mist.",
        "Watercolor bled softly where damp paper met the brush tip.",
        "折伞的人停在路口，雨丝斜织过街灯与砖墙的交界处。",
        "Potters center clay on the wheel before pulling walls thin and even.",
        "Saffron threads stained the rice gold while cumin scented the kitchen air.",
        "Skaters traced slow circles as the rink lights flickered at closing time.",
        "旧书页在阁楼里泛黄，批注墨迹与蠹虫细痕叠在同一段落旁。",
    ]
    return stem, life


def _topic_leaf_data(num_leaves: int) -> tuple[list[str], np.ndarray, dict[str, tuple[int, int]]]:
    """
    ``num_leaves`` must be ``2 * n_half`` with ``n_half`` in ``{4, 8}`` (8 or 16 leaves).
    Leaf order: ``stem[0..n_half-1]`` then ``life[0..n_half-1]``.
    """
    stem_full, life_full = _stem_and_life_pools()
    if num_leaves not in (8, 16):
        raise ValueError("num_leaves must be 8 or 16 (extend pools to support 32+).")
    n_half = num_leaves // 2
    stem = stem_full[:n_half]
    life = life_full[:n_half]
    texts = stem + life
    y = np.array([0] * n_half + [1] * n_half, dtype=np.int64)
    meta: dict[str, tuple[int, int]] = {}
    for i, t in enumerate(stem):
        meta[t] = (0, i)
    for i, t in enumerate(life):
        meta[t] = (1, i)
    return texts, y, meta


def _heldout_indices_by_template(
    root,
    meta: dict[str, tuple[int, int]],
    heldout_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Test = paths whose leaf template index is in the last ``heldout_per_class`` ids per class."""
    paths_rows = list(iter_root_leaf_paths(root))
    n_half = len([t for t in meta if meta[t][0] == 0])
    hi = set(range(n_half - heldout_per_class, n_half))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, path in enumerate(paths_rows):
        txt = path[-1].text
        cls, tid = meta[txt]
        (test_idx if tid in hi else train_idx).append(i)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


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
) -> dict[str, nn.Module]:
    readers: dict[str, nn.Module] = {
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


def _train_binary_head(
    reader: nn.Module,
    x: torch.Tensor,
    y01: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[float, float]:
    """Train ``Linear(dim,1)`` on reader output; optimize **reader + head** together."""
    dim = x.shape[2]
    head = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(list(reader.parameters()) + list(head.parameters()), lr=lr)
    reader.train()
    yf = y01.float()
    tr = train_idx.to(device)
    te = test_idx.to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = head(reader(x)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits[tr], yf[tr])
        loss.backward()
        opt.step()

    reader.eval()
    with torch.no_grad():
        logits = head(reader(x)).squeeze(-1)
        pred = (torch.sigmoid(logits) >= 0.5).long()
        train_acc = (pred[tr] == y01[tr]).float().mean().item()
        test_acc = (pred[te] == y01[te]).float().mean().item()
    return float(train_acc), float(test_acc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init-seed",
        type=int,
        default=42,
        help="torch seed before constructing each fresh reader stack (ridge vs train blocks)",
    )
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-leaves", type=int, default=16, help="8 or 16 (balanced fanout=2)")
    p.add_argument(
        "--leaf-split",
        type=str,
        choices=("heldout", "sample"),
        default="heldout",
        help="heldout: last K templates per class only in test (see --heldout-per-class); sample: stratified random paths",
    )
    p.add_argument(
        "--heldout-per-class",
        type=int,
        default=2,
        help="with heldout: number of leaf template indices per class reserved for test (last ids)",
    )
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--mamba-hidden", type=int, default=128)
    p.add_argument("--test-frac", type=float, default=0.25, help="only for leaf-split sample")
    p.add_argument("--ridge-lambda", type=float, default=1e-2)
    p.add_argument("--train-steps", type=int, default=0, help="if >0, train reader+linear head on train_idx only")
    p.add_argument("--train-lr", type=float, default=3e-3)
    p.add_argument("--no-mamba2", action="store_true")
    p.add_argument("--out-json", type=str, default="", help="write JSON (UTF-8)")
    args = p.parse_args()

    nl = args.n_leaves
    if nl not in (8, 16) or (math.log(nl) / math.log(args.fanout)) % 1 > 1e-9:
        print("ERROR: n-leaves must be 8 or 16 with fanout=2 (balanced).", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    leaf_texts, y, meta = _topic_leaf_data(nl)
    n_half = nl // 2
    if args.heldout_per_class < 1 or args.heldout_per_class >= n_half:
        print("ERROR: heldout-per-class must be in [1, n_half-1].", file=sys.stderr)
        return 1

    root = build_bottom_up_text_tree(
        leaf_texts,
        args.fanout,
        args.chunk_len,
        args.dim,
        device,
        torch.float32,
    )
    paths, n_paths = batched_paths(root)
    if n_paths != nl:
        raise RuntimeError(f"expected {nl} paths, got {n_paths}")

    if args.leaf_split == "heldout":
        train_idx, test_idx = _heldout_indices_by_template(root, meta, args.heldout_per_class)
    else:
        train_idx, test_idx = _stratified_indices(y, args.test_frac, args.seed + 1)

    if len(train_idx) < 4 or len(test_idx) < 2:
        print("ERROR: train/test too small; adjust heldout or n-leaves.", file=sys.stderr)
        return 1
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        print("ERROR: missing class in train or test split.", file=sys.stderr)
        return 1

    y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    include_mamba = mamba2_path_reader_available() and not args.no_mamba2

    def _probe_matrix(zp: np.ndarray) -> dict[str, float | int]:
        w = _ridge_weights(zp[train_idx], y_pm[train_idx], args.ridge_lambda)
        return {
            "train_acc": _accuracy_from_w(zp[train_idx], y_pm[train_idx], w),
            "test_acc": _accuracy_from_w(zp[test_idx], y_pm[test_idx], w),
            "output_dim": int(zp.shape[1]),
        }

    results_ridge: dict[str, object] = {}
    torch.manual_seed(args.init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.init_seed)

    x = paths.to(device)
    y01 = torch.tensor(y, dtype=torch.long, device=device)
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    test_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        raw_mean = x.mean(dim=1).detach().float().cpu().numpy()
        results_ridge["baseline_raw_mean_pool"] = _probe_matrix(raw_mean)

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
    with torch.no_grad():
        for name, model in readers.items():
            model.eval()
            z = model(x)
            zp = z.detach().float().cpu().numpy()
            results_ridge[name] = _probe_matrix(zp)

    trained: dict[str, object] | None = None
    if args.train_steps > 0:
        trained = {}
        torch.manual_seed(args.init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.init_seed)
        readers_t = _make_readers(
            dim=args.dim,
            nhead=args.nhead,
            tf_layers=args.tf_layers,
            gru_layers=args.gru_layers,
            mamba_layers=args.mamba_layers,
            mamba_hidden=args.mamba_hidden,
            include_mamba=include_mamba,
            device=device,
        )
        for name, model in readers_t.items():
            tr_acc, te_acc = _train_binary_head(
                model,
                x,
                y01,
                train_t,
                test_t,
                steps=args.train_steps,
                lr=args.train_lr,
                device=device,
            )
            trained[name] = {
                "train_acc": tr_acc,
                "test_acc": te_acc,
                "train_steps": args.train_steps,
                "train_lr": args.train_lr,
            }

    payload: dict[str, object] = {
        "kind": "probe_path_reader_linear",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "torch_version": torch.__version__,
        "device": str(device),
        "label": f"topic: {n_half} STEM + {n_half} life/arts leaf templates; y=0 first half / y=1 second half in leaf list",
        "leaf_split": args.leaf_split,
        "heldout_per_class": args.heldout_per_class if args.leaf_split == "heldout" else None,
        "tree": {
            "num_leaves": nl,
            "fanout": args.fanout,
            "chunk_len": args.chunk_len,
            "dim": args.dim,
            "path_shape": list(paths.shape),
        },
        "test_frac": args.test_frac if args.leaf_split == "sample" else None,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "ridge_lambda": args.ridge_lambda,
        "init_seed": args.init_seed,
        "ridge_untrained": results_ridge,
        "notes": (
            "ridge_untrained: linear classifier in weight space (ridge), not BCE head. "
            "trained: separate reader re-init + BCE on train indices only. "
            "Not GPT-2; not retrieval-head ID."
        ),
    }
    if trained is not None:
        payload["bce_reader_train"] = trained

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
