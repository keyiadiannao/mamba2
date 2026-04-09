#!/usr/bin/env python3
"""
**B-S2 延伸（本地）**：在 **path-batch 同 harness** 上，对 **path reader 输出向量**（非 HF 因果 LM）
做岭二分类，标签来自 **叶文本粗主题**（STEM vs 生活/文艺），与 ``probe_retrieval_correlation.py``（GPT-2 隐状态）
**并列对照**——接近「Mamba 路径表征里有什么可读信息」的第一步（仍 **不是** 2404 式 retrieval heads）。

  conda run -n mamba2 python scripts/research/probe_path_reader_linear.py --cpu --out-json results/metrics/probe_path_reader_linear_text8_cpu.json

依赖：``torch``、``transformers``（Mamba2 reader）、``numpy``。
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
from src.rag_tree.readers import GRUPathReader, Mamba2PathReader, TransformerPathReader, mamba2_path_reader_available
from src.rag_tree.tree import batched_paths


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


def _leaf_texts_topic8() -> tuple[list[str], np.ndarray]:
    """8 叶 fanout=2 depth=3：前 4 叶 STEM，后 4 叶生活/文艺；与 path 顺序一致（左到右叶序）。"""
    stem = [
        "Gradient descent minimizes loss over model parameters.",
        "Attention maps queries keys and values across token positions.",
        "Layer normalization rescales activations per token vector.",
        "KV caches avoid recomputing keys for past decoder steps.",
    ]
    life = [
        "The bread rose overnight under a linen cloth in a cool kitchen.",
        "Rowers feather their oars as the shell glides toward dawn mist.",
        "Watercolor bled softly where damp paper met the brush tip.",
        "折伞的人停在路口，雨丝斜织过街灯与砖墙的交界处。",
    ]
    texts = stem + life
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    return texts, y


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init-seed",
        type=int,
        default=42,
        help="torch seed for reader weight init (reproducible untrained forward)",
    )
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--mamba-hidden", type=int, default=128)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--ridge-lambda", type=float, default=1e-2)
    p.add_argument("--no-mamba2", action="store_true")
    p.add_argument("--out-json", type=str, default="", help="write JSON (UTF-8)")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.init_seed)

    leaf_texts, y = _leaf_texts_topic8()
    root = build_bottom_up_text_tree(
        leaf_texts,
        args.fanout,
        args.chunk_len,
        args.dim,
        device,
        torch.float32,
    )
    paths, n_paths = batched_paths(root)
    if n_paths != 8:
        raise RuntimeError(f"expected 8 paths, got {n_paths}")
    y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    train_idx, test_idx = _stratified_indices(y, args.test_frac, args.seed + 1)

    include_mamba = mamba2_path_reader_available() and not args.no_mamba2

    readers: dict[str, torch.nn.Module] = {
        "transformer": TransformerPathReader(
            dim=args.dim, nhead=args.nhead, num_layers=args.tf_layers
        ),
        "gru": GRUPathReader(dim=args.dim, num_layers=args.gru_layers),
    }
    if include_mamba:
        readers["mamba2"] = Mamba2PathReader(
            dim=args.dim,
            mamba_hidden=args.mamba_hidden,
            num_layers=args.mamba_layers,
        )

    results: dict[str, object] = {}

    def _probe_matrix(zp: np.ndarray) -> dict[str, float | int]:
        w = _ridge_weights(zp[train_idx], y_pm[train_idx], args.ridge_lambda)
        return {
            "train_acc": _accuracy_from_w(zp[train_idx], y_pm[train_idx], w),
            "test_acc": _accuracy_from_w(zp[test_idx], y_pm[test_idx], w),
            "output_dim": int(zp.shape[1]),
        }

    with torch.no_grad():
        x = paths.to(device)
        raw_mean = x.mean(dim=1).detach().float().cpu().numpy()
        results["baseline_raw_mean_pool"] = _probe_matrix(raw_mean)

        for name, model in readers.items():
            model.eval()
            model.to(device)
            z = model(x)
            zp = z.detach().float().cpu().numpy()
            results[name] = _probe_matrix(zp)

    payload = {
        "kind": "probe_path_reader_linear",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "torch_version": torch.__version__,
        "device": str(device),
        "label": "topic8_leaf: first 4 leaves STEM templates, last 4 life/arts templates",
        "tree": {
            "num_leaves": 8,
            "fanout": args.fanout,
            "chunk_len": args.chunk_len,
            "dim": args.dim,
            "path_shape": list(paths.shape),
        },
        "test_frac": args.test_frac,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "ridge_lambda": args.ridge_lambda,
        "init_seed": args.init_seed,
        "readers": results,
        "notes": (
            "Readers are randomly initialized (same as cold start); compare to baseline_raw_mean_pool on hash embeddings. "
            "Trained readers would be a separate experiment. Not GPT-2; not retrieval-head identification."
        ),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
