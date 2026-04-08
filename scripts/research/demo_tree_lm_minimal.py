#!/usr/bin/env python3
"""
真 LM + 生成头 + 文本形树上的**最小闭环**：

1. 自底向上建树（``build_bottom_up_text_tree``），叶行为 UTF-8 文本；
2. 每条根—叶路径将节点 ``text`` 拼成文档；
3. ``AutoModelForCausalLM``：逐路径 **teacher-forcing CE** + 任选一条路径 **贪心续写**；
4. 可选 ``--train-one-step``：对当前所有路径文档的 **平均 loss** 做一次 **AdamW** 更新。

默认小模型 ``sshleifer/tiny-gpt2``（便于本机/5060）；可改 ``--model gpt2`` 等。

  python scripts/research/demo_tree_lm_minimal.py --cpu
  python scripts/research/demo_tree_lm_minimal.py --train-one-step --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.tree_lm_closure import (
    causal_lm_mean_loss_on_documents,
    ensure_causal_lm_tokenizer,
    generate_continuation,
    iter_path_documents,
    train_one_step_mean_loss,
)

# 与 ``benchmark_text_tree.py`` 默认叶一致（``scripts`` 非包，避免跨目录 import）
_DEFAULT_LEAF_LINES = [
    "RAG combines retrieval with generation.",
    "Tree RAG uses hierarchical summaries.",
    "Mamba uses a recurrent state for long sequences.",
    "Transformers use quadratic attention in length.",
    "状态空间模型压缩历史为固定维度。",
    "检索头预测是否需要查外部知识。",
    "实验在5060与AutoDL上分工运行。",
    "复现依赖Git与环境锁定文件。",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2", help="HF causal LM id")
    p.add_argument("--leaf-file", type=str, default="", help="UTF-8 一叶一行；默认用与 benchmark_text_tree 相同的内置 8 句")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128, help="仅用于树占位嵌入；LM 走 tokenizer，与此维无关")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-length", type=int, default=256, help="tokenizer 截断长度")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--path-index", type=int, default=0, help="对第几条根—叶路径做续写（0-based）")
    p.add_argument("--train-one-step", action="store_true", help="对全部路径文档平均 CE 做一次 optimizer.step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-json", type=str, default="", help="可选：写入本次指标 JSON")
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers", file=sys.stderr)
        return 1

    def _load_pretrained():
        tok = AutoTokenizer.from_pretrained(args.model)
        m = AutoModelForCausalLM.from_pretrained(args.model)
        return tok, m

    try:
        tok, model = _load_pretrained()
    except OSError as e:
        print("ERROR: cannot load tokenizer/model:", e, file=sys.stderr)
        print(
            "Hint: need Hugging Face Hub access (or mirror via HF_ENDPOINT). "
            "Or pass --model path\\to\\local\\checkpoint with config.json + weights.",
            file=sys.stderr,
        )
        return 1

    if args.leaf_file:
        raw = Path(args.leaf_file).read_text(encoding="utf-8")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        lines = list(_DEFAULT_LEAF_LINES)

    n = len(lines)
    f = args.fanout
    cur = 1
    depth_edges = 0
    while cur < n:
        cur *= f
        depth_edges += 1
    if cur != n:
        print(
            f"ERROR: leaf count {n} must equal fanout**depth (fanout={f}); nearest power is {cur}.",
            file=sys.stderr,
        )
        return 1

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    root = build_bottom_up_text_tree(lines, f, args.chunk_len, args.dim, dev, torch.float32)

    ensure_causal_lm_tokenizer(tok)
    model.to(dev)
    model.eval()

    pairs = iter_path_documents(root)
    documents = [doc for _, doc in pairs]

    mean_loss, per = causal_lm_mean_loss_on_documents(
        model, tok, documents, dev, max_length=args.max_length
    )
    print("paths", len(documents), "mean_nll_loss", float(mean_loss.cpu()), "per_path", [float(x.cpu()) for x in per])

    pi = max(0, min(args.path_index, len(documents) - 1))
    prefix = documents[pi]
    cont = generate_continuation(
        model,
        tok,
        prefix,
        dev,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_length,
    )
    print("generate_path_index", pi)
    print("continuation", cont[: 600] + ("…" if len(cont) > 600 else ""))

    train_loss_out: float | None = None
    mean_after: float | None = None
    if args.train_one_step:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_loss_out = train_one_step_mean_loss(
            model, opt, tok, documents, dev, max_length=args.max_length
        )
        model.eval()
        ma, _ = causal_lm_mean_loss_on_documents(
            model, tok, documents, dev, max_length=args.max_length
        )
        mean_after = float(ma.cpu())
        print("train_one_step_mean_loss", train_loss_out, "mean_nll_loss_after", mean_after)

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        payload = {
            "kind": "tree_lm_minimal_demo",
            "model": args.model,
            "device": str(dev),
            "num_paths": len(documents),
            "mean_nll_loss": float(mean_loss.cpu()),
            "per_path_nll_loss": [float(x.cpu()) for x in per],
            "generate_path_index": pi,
            "continuation_preview": cont[:500],
            "train_one_step": bool(args.train_one_step),
            "train_one_step_mean_loss": train_loss_out,
            "mean_nll_loss_after": mean_after,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("wrote", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
