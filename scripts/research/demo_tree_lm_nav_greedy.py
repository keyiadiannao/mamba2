#!/usr/bin/env python3
"""
**树上导航任务（启发式）**：在每个内部节点，用因果 LM 对「当前 walk + 子节点」整段文档算 CE，
取 **loss 最小**的子节点下降；与 **金路径**（``iter_root_leaf_paths`` 的第 ``k`` 条叶）逐步对比。

默认模型 ``sshleifer/tiny-gpt2``、默认 8 叶平衡树（与 ``demo_tree_lm_minimal.py`` 一致）。
**非**训练好的策略；指标用于 pipeline 与「随机/弱基线」对照。

  python scripts/research/demo_tree_lm_nav_greedy.py --cpu --target-leaf-index 7
  python scripts/research/demo_tree_lm_nav_greedy.py --cpu --eval-all-leaves --out-json results/metrics/tree_lm_nav_greedy_eval.json
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
from src.rag_tree.tree_lm_nav_eval import GreedyLmNavResult, greedy_navigate_by_lm_child_loss

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
                "losses_per_child": s.losses_per_child,
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
    p.add_argument("--target-leaf-index", type=int, default=7, help="金叶下标（0..num_leaves-1）")
    p.add_argument(
        "--eval-all-leaves",
        action="store_true",
        help="对每条叶各跑一次贪心导航，汇总 reach_rate 与平均 child_choice_accuracy",
    )
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
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except OSError as e:
        print("ERROR: cannot load model:", e, file=sys.stderr)
        return 1

    ensure_causal_lm_tokenizer(tok)
    model.to(dev)
    model.eval()

    root = build_bottom_up_text_tree(lines, f, args.chunk_len, args.dim, dev, torch.float32)
    num_leaves = len(list(iter_root_leaf_paths(root)))

    if args.eval_all_leaves:
        results: list[GreedyLmNavResult] = []
        for k in range(num_leaves):
            results.append(
                greedy_navigate_by_lm_child_loss(
                    root,
                    k,
                    model,
                    tok,
                    dev,
                    max_length=args.max_length,
                )
            )
        reach_rate = sum(1.0 for r in results if r.reached_target_leaf) / len(results)
        mean_acc = sum(r.child_choice_accuracy for r in results) / len(results)
        print("num_leaves", num_leaves, "reach_rate", reach_rate, "mean_child_choice_accuracy", mean_acc)
        for r in results:
            print(
                f"  leaf{r.target_leaf_index} reached={r.reached_target_leaf} acc={r.child_choice_accuracy:.4f}",
            )
        payload = {
            "kind": "tree_lm_nav_greedy_eval",
            "git_sha": _git_short_sha(_REPO_ROOT),
            "model": args.model,
            "device": str(dev),
            "eval_all_leaves": True,
            "num_leaves": num_leaves,
            "reach_rate": reach_rate,
            "mean_child_choice_accuracy": mean_acc,
            "per_leaf": [_result_to_dict(r) for r in results],
        }
    else:
        if args.target_leaf_index < 0 or args.target_leaf_index >= num_leaves:
            print(f"ERROR: target_leaf_index must be in [0,{num_leaves})", file=sys.stderr)
            return 1
        r = greedy_navigate_by_lm_child_loss(
            root,
            args.target_leaf_index,
            model,
            tok,
            dev,
            max_length=args.max_length,
        )
        print("target_leaf_index", r.target_leaf_index)
        print("reached_target_leaf", r.reached_target_leaf)
        print("child_choice_accuracy", r.child_choice_accuracy)
        print("steps", [s.__dict__ for s in r.steps])
        payload = {
            "kind": "tree_lm_nav_greedy",
            "git_sha": _git_short_sha(_REPO_ROOT),
            "model": args.model,
            "device": str(dev),
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
