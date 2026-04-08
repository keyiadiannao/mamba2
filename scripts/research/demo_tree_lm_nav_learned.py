#!/usr/bin/env python3
"""
**目标叶条件**可学习子指针：冻结 ``tiny-gpt2``，最后 token 隐状态 + **goal 叶下标嵌入** → 子节点 logits；
监督为 **CrossEntropy**（每 epoch 对 24 条样本累积梯度后一步 AdamW）。

同一前缀在不同目标叶下金孩子不同，故必须 **条件化 goal**（非盲导航）。

与 **X-20260423** 启发式（整段「walk+子」LM CE）对照：默认 8 叶上 **reach_rate** 约 **0.375** vs **0.125**。

  python scripts/research/demo_tree_lm_nav_learned.py --cpu --epochs 250 --eval-all-leaves
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

import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.tree import iter_root_leaf_paths
from src.rag_tree.tree_lm_closure import ensure_causal_lm_tokenizer
from src.rag_tree.tree_lm_nav_eval import GreedyLmNavResult
from src.rag_tree.tree_lm_nav_learned import (
    GoalConditionedChildHead,
    greedy_navigate_by_child_head,
    iter_goal_conditioned_examples,
    max_fanout_of_tree,
    train_child_head,
)

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


def _result_to_dict(r: GreedyLmNavResult) -> dict:
    return {
        "target_leaf_index": r.target_leaf_index,
        "reached_target_leaf": r.reached_target_leaf,
        "num_internal_decisions": r.num_internal_decisions,
        "child_choice_accuracy": r.child_choice_accuracy,
        "steps": [
            {
                "depth": s.depth,
                "scores_per_child": s.losses_per_child,
                "chosen_child": s.chosen_child,
                "gold_child": s.gold_child,
                "correct": s.correct,
            }
            for s in r.steps
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--leaf-file", type=str, default="")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0, help="每 epoch 打乱训练样本的 RNG 种子")
    p.add_argument("--freeze-lm", action="store_true", default=True)
    p.add_argument("--no-freeze-lm", action="store_false", dest="freeze_lm")
    p.add_argument("--goal-leaf-index", type=int, default=7)
    p.add_argument("--eval-all-leaves", action="store_true")
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers", file=sys.stderr)
        return 1

    if args.leaf_file:
        raw = Path(args.leaf_file).read_text(encoding="utf-8")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        lines = list(_DEFAULT_LEAF_LINES)

    n = len(lines)
    f = args.fanout
    cur = 1
    while cur < n:
        cur *= f
    if cur != n:
        print(f"ERROR: leaf count {n} must equal fanout**depth (fanout={f}).", file=sys.stderr)
        return 1

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        lm = AutoModelForCausalLM.from_pretrained(args.model)
    except OSError as e:
        print("ERROR: cannot load model:", e, file=sys.stderr)
        return 1

    ensure_causal_lm_tokenizer(tok)
    lm.to(dev)

    root = build_bottom_up_text_tree(lines, f, args.chunk_len, args.dim, dev, torch.float32)
    num_leaves = len(list(iter_root_leaf_paths(root)))
    mf = max_fanout_of_tree(root)
    examples = iter_goal_conditioned_examples(root)

    hs = int(lm.config.hidden_size)
    head = GoalConditionedChildHead(
        lm_hidden_size=hs,
        num_leaves=num_leaves,
        max_fanout=mf,
        goal_dim=32,
    ).to(dev)

    loss_curve = train_child_head(
        lm,
        tok,
        head,
        examples,
        dev,
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        freeze_lm=args.freeze_lm,
        max_fanout=mf,
        shuffle_seed=args.seed,
    )
    print("train_epochs", args.epochs, "final_epoch_mean_loss", loss_curve[-1] if loss_curve else None)

    if args.eval_all_leaves:
        results: list[GreedyLmNavResult] = []
        for k in range(num_leaves):
            results.append(
                greedy_navigate_by_child_head(
                    root,
                    k,
                    lm,
                    tok,
                    head,
                    dev,
                    max_length=args.max_length,
                    max_fanout=mf,
                )
            )
        reach_rate = sum(1.0 for r in results if r.reached_target_leaf) / len(results)
        mean_acc = sum(r.child_choice_accuracy for r in results) / len(results)
        print("num_leaves", num_leaves, "reach_rate", reach_rate, "mean_child_choice_accuracy", mean_acc)
        for r in results:
            print(f"  leaf{r.target_leaf_index} reached={r.reached_target_leaf} acc={r.child_choice_accuracy:.4f}")
        payload = {
            "kind": "tree_lm_nav_learned_eval",
            "git_sha": _git_short_sha(_REPO_ROOT),
            "model": args.model,
            "device": str(dev),
            "goal_conditioned": True,
            "freeze_lm": args.freeze_lm,
            "epochs": args.epochs,
            "lr": args.lr,
            "shuffle_seed": args.seed,
            "train_examples": len(examples),
            "final_epoch_mean_loss": loss_curve[-1] if loss_curve else None,
            "loss_curve_last5": loss_curve[-5:] if len(loss_curve) >= 5 else loss_curve,
            "eval_all_leaves": True,
            "num_leaves": num_leaves,
            "reach_rate": reach_rate,
            "mean_child_choice_accuracy": mean_acc,
            "per_leaf": [_result_to_dict(r) for r in results],
        }
    else:
        if args.goal_leaf_index < 0 or args.goal_leaf_index >= num_leaves:
            print(f"ERROR: goal_leaf_index must be in [0,{num_leaves})", file=sys.stderr)
            return 1
        r = greedy_navigate_by_child_head(
            root,
            args.goal_leaf_index,
            lm,
            tok,
            head,
            dev,
            max_length=args.max_length,
            max_fanout=mf,
        )
        print("goal_leaf_index", r.target_leaf_index, "reached", r.reached_target_leaf, "acc", r.child_choice_accuracy)
        payload = {
            "kind": "tree_lm_nav_learned",
            "git_sha": _git_short_sha(_REPO_ROOT),
            "model": args.model,
            "device": str(dev),
            "goal_conditioned": True,
            "freeze_lm": args.freeze_lm,
            "epochs": args.epochs,
            "lr": args.lr,
            "shuffle_seed": args.seed,
            "train_examples": len(examples),
            "final_epoch_mean_loss": loss_curve[-1] if loss_curve else None,
            "eval_all_leaves": False,
            **_result_to_dict(r),
        }

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("wrote", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
