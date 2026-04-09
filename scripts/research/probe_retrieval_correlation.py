#!/usr/bin/env python3
"""
B-S2：对 **HF 小因果 LM** 各层隐状态做 **二分类线性探针**（岭回归闭式解，NumPy only）。

目的不是宣称「已发现检索头」，而是提供 **可复现脚手架**：同一批文本上比较 **结构化标签**
（合成 **marker** 子串）与 **随机标签** 的 **test 准确率**；若前者显著高于后者，说明隐状态
**携带** 与标签相关的线性可读信息（文献中常见探针设定；与 **path-batch 延迟** 不可混读）。

  conda run -n mamba2 python scripts/research/probe_retrieval_correlation.py --cpu --out-json results/metrics/probe_retrieval_linear_demo.json

依赖：``torch``、``transformers``、``numpy``（无需 ``sklearn``）。

若本机访问 ``huggingface.co`` **SSL 失败**，可设镜像（与 **`AUTODL_SETUP`** 一致）或依赖本地缓存权重；探针在模型已缓存时仍可跑通。
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
    """y in {-1, 1}; minimize ||Xw - y||^2 + lam||w||^2."""
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


def _build_texts_marker(n_samples: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Balanced binary: class 1 iff substring ``RETRVPROBE`` appears."""
    rng = np.random.default_rng(seed)
    bases = [
        "RAG combines retrieval with generation on long contexts.",
        "Tree indices help structure navigation over documents.",
        "Mamba state tracks history with fixed memory per step.",
        "Attention scores route information between token positions.",
        "状态空间模型压缩历史为固定维度的向量。",
        "实验脚本在本地与云端分工运行以节省机时。",
        "The optimizer adjusts weights using gradient descent steps.",
        "Caching avoids recomputation when revisiting earlier segments.",
    ]

    def with_marker(s: str) -> str:
        parts = s.split()
        if len(parts) < 2:
            return s + " RETRVPROBE"
        j = 1 + int(rng.integers(0, len(parts) - 1))
        parts.insert(j, "RETRVPROBE")
        return " ".join(parts)

    half = n_samples // 2
    texts: list[str] = []
    labels: list[int] = []
    for _ in range(half):
        s = str(rng.choice(bases))
        texts.append(s)
        labels.append(0)
    for _ in range(n_samples - half):
        s = with_marker(str(rng.choice(bases)))
        texts.append(s)
        labels.append(1)
    perm = rng.permutation(len(texts))
    y = np.array([labels[i] for i in perm], dtype=np.int64)
    texts = [texts[i] for i in perm]
    return texts, y


def _build_texts_digit(n_samples: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Balanced: class 1 iff substring contains an ASCII digit (synthetic)."""
    rng = np.random.default_rng(seed)
    bases = [
        "Short context about rivers forests and hills.",
        "Neural networks learn from labeled training examples.",
        "The cache stores intermediate activations for reuse.",
        "向量空间中的点积衡量方向相似程度。",
    ]
    half = n_samples // 2
    texts: list[str] = []
    labels: list[int] = []
    for _ in range(half):
        texts.append(str(rng.choice(bases)))
        labels.append(0)
    for _ in range(n_samples - half):
        b = str(rng.choice(bases))
        # inject digit token (not in bases)
        words = b.split()
        j = int(rng.integers(0, len(words) + 1))
        words.insert(j, "42")
        texts.append(" ".join(words))
        labels.append(1)
    perm = rng.permutation(len(texts))
    y = np.array([labels[i] for i in perm], dtype=np.int64)
    texts = [texts[i] for i in perm]
    return texts, y


def _pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> np.ndarray:
    """hidden: (batch, seq, dim); mask: (batch, seq). Mean pool over valid tokens."""
    h = hidden.detach().float().cpu().numpy()
    m = attention_mask.detach().float().cpu().numpy()
    m = m[:, :, np.newaxis]
    s = (h * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-6, None)
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=160, help="balanced classes when using marker")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--ridge-lambda", type=float, default=1e-2)
    p.add_argument(
        "--label-mode",
        type=str,
        choices=("marker", "digit", "random"),
        default="marker",
        help="marker: synthetic RETRVPROBE; digit: any digit in text; random: shuffle labels (control)",
    )
    p.add_argument("--no-random-control", action="store_true", help="skip second pass with shuffled labels")
    p.add_argument("--out-json", type=str, default="", help="write metrics JSON (UTF-8)")
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to(device)

    if args.label_mode == "digit":
        texts, y = _build_texts_digit(args.n_samples, args.seed)
    else:
        texts, y = _build_texts_marker(args.n_samples, args.seed)
    if args.label_mode == "random":
        rng = np.random.default_rng(args.seed + 99)
        rng.shuffle(y)

    if len(np.unique(y)) < 2:
        print("ERROR: single class after label construction; increase n-samples or change mode", file=sys.stderr)
        return 1

    n_layers_total: int | None = None
    per_layer_h: list[list[np.ndarray]] = []

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if n_layers_total is None:
                n_layers_total = len(hs)
                per_layer_h = [[] for _ in range(n_layers_total)]
            for li, h in enumerate(hs):
                pooled = _pool_hidden(h, enc["attention_mask"])
                per_layer_h[li].append(pooled)

    if n_layers_total is None:
        return 1

    X_layers = [np.concatenate(chunks, axis=0) for chunks in per_layer_h]
    y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float64)

    train_idx, test_idx = _stratified_indices(y, args.test_frac, args.seed + 1)

    def run_probe(X: np.ndarray, y_signed: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y_signed[tr], y_signed[te]
        w = _ridge_weights(Xtr, ytr, args.ridge_lambda)
        return {
            "train_acc": _accuracy_from_w(Xtr, ytr, w),
            "test_acc": _accuracy_from_w(Xte, yte, w),
        }

    layers_out: list[dict] = []
    for li, X in enumerate(X_layers):
        m = run_probe(X, y_pm, train_idx, test_idx)
        layers_out.append(
            {
                "layer_index": li,
                "hidden_dim": int(X.shape[1]),
                **m,
            }
        )

    random_layers_out: list[dict] | None = None
    if not args.no_random_control and args.label_mode != "random":
        rng = np.random.default_rng(args.seed + 42)
        y_rand = y.copy()
        rng.shuffle(y_rand)
        y_rand_pm = np.where(y_rand == 1, 1.0, -1.0).astype(np.float64)
        random_layers_out = []
        for li, X in enumerate(X_layers):
            m = run_probe(X, y_rand_pm, train_idx, test_idx)
            random_layers_out.append(
                {
                    "layer_index": li,
                    "hidden_dim": int(X.shape[1]),
                    **m,
                }
            )

    payload: dict = {
        "kind": "probe_retrieval_correlation",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "torch_version": torch.__version__,
        "model": args.model,
        "device": str(device),
        "label_mode": args.label_mode,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "test_frac": args.test_frac,
        "ridge_lambda": args.ridge_lambda,
        "layers": layers_out,
    }
    if random_layers_out is not None:
        payload["random_label_control"] = {"layers": random_layers_out}

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
