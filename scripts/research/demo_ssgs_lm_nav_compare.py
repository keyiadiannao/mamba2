#!/usr/bin/env python3
"""
**SSGS×LM 玩具对照**：与 **X-20260424** 相同的 **8 句叶文本树**（``dim=128``、``chunk_len=8``），对每个 **goal 叶**：

- **SSGS**：``dfs_ssgs_mamba`` + ``MambaNavState``（**必达**该叶），记录 **snapshots / rollbacks / leaf_checks**；
- **LM**：冻结 ``tiny-gpt2`` + **goal 子头**（同 ``demo_tree_lm_nav_learned.py`` 训练），**贪心**下降（**可能未达**）。

二者 **任务不同**（DFS 试错 vs 每步 argmax），JSON 仅作 **并列归档**，不可混为同一 harness。登记 **X-20260425-ssgs-lm-nav-compare**。

  python scripts/research/demo_ssgs_lm_nav_compare.py --cpu --epochs 250 --out-json results/metrics/ssgs_lm_nav_compare_default8_cpu.json
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
from src.rag_tree.ssgs import (
    MambaNavState,
    SSGSTrace,
    build_toy_mamba2_for_ssgs,
    clear_tree_snapshots,
    dfs_ssgs_mamba,
    mount_mamba_cache_meta_on_tree,
)
from src.rag_tree.tree import iter_root_leaf_paths
from src.rag_tree.tree_lm_closure import ensure_causal_lm_tokenizer
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


def _run_ssgs_for_goal(
    root,
    target_leaf,
    model,
    *,
    mount_snapshot,
) -> tuple[bool, SSGSTrace]:
    clear_tree_snapshots(root)
    state = MambaNavState(model=model)
    tr = SSGSTrace()

    def leaf_goal(n: object) -> bool:
        return n is target_leaf

    ok = dfs_ssgs_mamba(
        root,
        state,
        leaf_goal=leaf_goal,
        trace=tr,
        mount_snapshot=mount_snapshot,
    )
    return ok, tr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-lm", action="store_true", default=True)
    p.add_argument("--no-freeze-lm", action="store_false", dest="freeze_lm")
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers", file=sys.stderr)
        return 1

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    lines = list(_DEFAULT_LEAF_LINES)
    n = len(lines)
    f = args.fanout
    cur = 1
    while cur < n:
        cur *= f
    if cur != n:
        print(f"ERROR: leaf count {n} must equal fanout**depth (fanout={f}).", file=sys.stderr)
        return 1

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        lm = AutoModelForCausalLM.from_pretrained(args.model)
    except OSError as e:
        print("ERROR: cannot load model:", e, file=sys.stderr)
        return 1

    ensure_causal_lm_tokenizer(tok)
    lm.to(dev)

    root = build_bottom_up_text_tree(lines, f, args.chunk_len, args.dim, dev, torch.float32)
    paths = list(iter_root_leaf_paths(root))
    leaves = [p[-1] for p in paths]
    num_leaves = len(leaves)
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
    print("train final_epoch_mean_loss", loss_curve[-1] if loss_curve else None)

    mamba = build_toy_mamba2_for_ssgs(args.dim, dev, num_layers=args.mamba_layers)

    per_goal: list[dict] = []
    for k, target in enumerate(leaves):
        ok_ssgs, tr = _run_ssgs_for_goal(
            root,
            target,
            mamba,
            mount_snapshot=mount_mamba_cache_meta_on_tree,
        )
        r_lm = greedy_navigate_by_child_head(
            root,
            k,
            lm,
            tok,
            head,
            dev,
            max_length=args.max_length,
            max_fanout=mf,
        )
        per_goal.append(
            {
                "goal_leaf_index": k,
                "ssgs_ok": ok_ssgs,
                "ssgs_snapshots_taken": tr.snapshots_taken,
                "ssgs_rollbacks": tr.rollbacks,
                "ssgs_leaf_checks": tr.leaf_checks,
                "ssgs_event_count": len(tr.events),
                "lm_reached_target_leaf": r_lm.reached_target_leaf,
                "lm_child_choice_accuracy": r_lm.child_choice_accuracy,
                "lm_num_internal_decisions": r_lm.num_internal_decisions,
            }
        )
        print(
            f"goal {k} ssgs_ok={ok_ssgs} snap={tr.snapshots_taken} rb={tr.rollbacks} "
            f"lm_reached={r_lm.reached_target_leaf} lm_acc={r_lm.child_choice_accuracy:.4f}"
        )

    lm_reach = sum(1 for x in per_goal if x["lm_reached_target_leaf"]) / len(per_goal)
    ssgs_ok_all = all(x["ssgs_ok"] for x in per_goal)
    lm_mean_acc = sum(x["lm_child_choice_accuracy"] for x in per_goal) / len(per_goal)

    payload = {
        "kind": "ssgs_lm_nav_compare",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "model_lm": args.model,
        "device": str(dev),
        "mamba_hidden": args.dim,
        "mamba_layers": args.mamba_layers,
        "mamba_torch_forward_only": dev.type == "cuda",
        "freeze_lm": args.freeze_lm,
        "epochs": args.epochs,
        "lr": args.lr,
        "shuffle_seed": args.seed,
        "train_examples": len(examples),
        "final_epoch_mean_loss": loss_curve[-1] if loss_curve else None,
        "num_leaves": num_leaves,
        "ssgs_all_goals_ok": ssgs_ok_all,
        "lm_reach_rate": lm_reach,
        "lm_mean_child_choice_accuracy": lm_mean_acc,
        "per_goal": per_goal,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("wrote", out_path)

    return 0 if ssgs_ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
