"""用 **因果 LM 的 teacher-forcing CE** 给每个子节点打分，**贪心**选 loss 最小的子（启发式，非训练好的策略）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch

from src.rag_tree.tree import TreeNode, iter_root_leaf_paths
from src.rag_tree.tree_lm_closure import causal_lm_loss_for_document, path_to_document


def gold_child_index(target_path: Sequence[TreeNode], node: TreeNode) -> int:
    """
    ``node`` 须在 ``target_path`` 上且为**内部节点**；返回通往目标叶的**下一子**在 ``node.children`` 中的下标。
    """
    if node.is_leaf():
        raise ValueError("gold_child_index: node is leaf")
    i = None
    for idx, p in enumerate(target_path):
        if p is node:
            i = idx
            break
    if i is None:
        raise ValueError("gold_child_index: node not on target_path (use object identity)")
    if i >= len(target_path) - 1:
        raise ValueError("gold_child_index: node is last on path (should be leaf)")
    nxt = target_path[i + 1]
    for idx, ch in enumerate(node.children):
        if ch is nxt:
            return idx
    raise RuntimeError("target_path does not continue through a child of node")


@dataclass
class GreedyLmNavStep:
    """一步贪心决策记录。"""

    depth: int
    losses_per_child: List[float]
    chosen_child: int
    gold_child: int  # 若当前节点已偏离金路径前缀，记为 -1
    correct: bool


@dataclass
class GreedyLmNavResult:
    target_leaf_index: int
    reached_target_leaf: bool
    steps: List[GreedyLmNavStep] = field(default_factory=list)

    @property
    def num_internal_decisions(self) -> int:
        return len(self.steps)

    @property
    def child_choice_accuracy(self) -> float:
        if not self.steps:
            return 1.0
        return sum(1.0 for s in self.steps if s.correct) / len(self.steps)


@torch.inference_mode()
def greedy_navigate_by_lm_child_loss(
    root: TreeNode,
    target_leaf_index: int,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    *,
    max_length: int = 512,
    sep: str = "\n\n",
) -> GreedyLmNavResult:
    """
    从根出发：在每个内部节点，对每个子 ``ch`` 计算
    ``path_to_document(path_to_current + [ch])`` 的 CE，取 **argmin** 下降。

    ``target_leaf_index``：与 ``iter_root_leaf_paths`` **同一顺序**（左先 DFS）的叶下标。
    """
    paths = list(iter_root_leaf_paths(root))
    if target_leaf_index < 0 or target_leaf_index >= len(paths):
        raise ValueError(f"target_leaf_index {target_leaf_index} out of range [0,{len(paths)})")
    target_path = paths[target_leaf_index]

    cur = root
    walk: List[TreeNode] = [cur]
    steps: List[GreedyLmNavStep] = []

    while not cur.is_leaf():
        depth = len(walk) - 1
        on_gold_prefix = cur is target_path[depth]
        if on_gold_prefix:
            gold = gold_child_index(target_path, cur)
        else:
            gold = -1

        losses: List[float] = []
        for ch in cur.children:
            doc = path_to_document(walk + [ch], sep=sep)
            ell = causal_lm_loss_for_document(
                model, tokenizer, doc, device, max_length=max_length
            )
            losses.append(float(ell.cpu()) if not torch.isnan(ell) else float("inf"))

        chosen = int(min(range(len(losses)), key=lambda j: losses[j]))
        correct = on_gold_prefix and (chosen == gold)
        steps.append(
            GreedyLmNavStep(
                depth=depth,
                losses_per_child=losses,
                chosen_child=chosen,
                gold_child=gold,
                correct=correct,
            )
        )
        cur = cur.children[chosen]
        walk.append(cur)

    reached = walk[-1] is target_path[-1]
    return GreedyLmNavResult(
        target_leaf_index=target_leaf_index,
        reached_target_leaf=reached,
        steps=steps,
    )
