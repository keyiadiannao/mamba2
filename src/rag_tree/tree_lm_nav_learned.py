"""**目标叶条件**子节点指针：LM 最后 token 隐状态 + ``goal_leaf_index`` 嵌入 → ``fanout`` 类。

同一内部节点在不同「要到达的叶」下金孩子不同，故必须 **条件化目标**（最小可学习闭环；非盲导航）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rag_tree.tree import TreeNode, iter_root_leaf_paths
from src.rag_tree.tree_lm_closure import path_to_document
from src.rag_tree.tree_lm_nav_eval import GreedyLmNavResult, GreedyLmNavStep, gold_child_index


@dataclass
class GoalConditionedNavExample:
    prefix_doc: str
    gold_child: int
    num_children: int
    goal_leaf_index: int


def iter_goal_conditioned_examples(root: TreeNode, *, sep: str = "\n\n") -> List[GoalConditionedNavExample]:
    """每条根—叶路径上每个内部节点一条监督（前缀 = 根到该节点文档）。"""
    out: List[GoalConditionedNavExample] = []
    paths = list(iter_root_leaf_paths(root))
    for leaf_idx, path in enumerate(paths):
        for d in range(len(path) - 1):
            node = path[d]
            if node.is_leaf():
                continue
            prefix = path_to_document(path[: d + 1], sep=sep)
            gold = gold_child_index(path, node)
            out.append(
                GoalConditionedNavExample(
                    prefix_doc=prefix,
                    gold_child=gold,
                    num_children=len(node.children),
                    goal_leaf_index=leaf_idx,
                )
            )
    return out


def _last_token_hidden(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    """因果 LM 最后一层、**最后一个非 pad token** 的隐向量 ``[H]``。"""
    if not (text or "").strip():
        raise ValueError("empty prefix for hidden state")
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]  # [1, T, H]
    attn = enc["attention_mask"]
    last_i = int(attn[0].sum().item()) - 1
    return h[0, last_i, :]


class GoalConditionedChildHead(nn.Module):
    """``[ h_lm || e_goal ] -> logits over max_fanout``（无效类在 loss / 推理时掩掉）。"""

    def __init__(
        self,
        *,
        lm_hidden_size: int,
        num_leaves: int,
        max_fanout: int,
        goal_dim: int = 32,
    ) -> None:
        super().__init__()
        self.goal_emb = nn.Embedding(num_leaves, goal_dim)
        self.proj = nn.Linear(lm_hidden_size + goal_dim, max_fanout)

    def forward(self, h_lm: torch.Tensor, goal_leaf_index: torch.Tensor) -> torch.Tensor:
        # h_lm: [B, H], goal: [B] int64
        g = self.goal_emb(goal_leaf_index)
        x = torch.cat([h_lm, g], dim=-1)
        return self.proj(x)


def train_child_head(
    lm: torch.nn.Module,
    tokenizer,
    head: GoalConditionedChildHead,
    examples: Sequence[GoalConditionedNavExample],
    device: torch.device,
    *,
    max_length: int,
    epochs: int,
    lr: float,
    freeze_lm: bool = True,
    max_fanout: int,
    shuffle_seed: int = 0,
) -> List[float]:
    """AdamW 更新 ``head``（可选解冻 LM）。每 epoch 对**全部**样本累积梯度后 **一步** ``optimizer.step()``（小数据更稳）。"""
    if freeze_lm:
        lm.eval()
        for p in lm.parameters():
            p.requires_grad = False
    else:
        for p in lm.parameters():
            p.requires_grad = True
        lm.train()

    head.train()
    opt = torch.optim.AdamW(
        list(head.parameters()) + ([] if freeze_lm else list(lm.parameters())),
        lr=lr,
    )
    losses: List[float] = []

    ex_list = [e for e in examples if e.num_children >= 1]
    rng = random.Random(shuffle_seed)
    for _ in range(epochs):
        rng.shuffle(ex_list)
        opt.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for ex in ex_list:
            if freeze_lm:
                with torch.inference_mode():
                    h = _last_token_hidden(lm, tokenizer, ex.prefix_doc, device, max_length=max_length)
            else:
                h = _last_token_hidden(lm, tokenizer, ex.prefix_doc, device, max_length=max_length)
            logits = head(h.unsqueeze(0), torch.tensor([ex.goal_leaf_index], device=device, dtype=torch.long))
            logits_s = logits[0, : ex.num_children]
            loss = F.cross_entropy(
                logits_s.unsqueeze(0),
                torch.tensor([ex.gold_child], device=device, dtype=torch.long),
            )
            (loss / len(ex_list)).backward()
            epoch_loss += float(loss.detach().cpu())
        opt.step()
        losses.append(epoch_loss / len(ex_list))

    return losses


@torch.inference_mode()
def greedy_navigate_by_child_head(
    root: TreeNode,
    target_leaf_index: int,
    lm: torch.nn.Module,
    tokenizer,
    head: GoalConditionedChildHead,
    device: torch.device,
    *,
    max_length: int,
    max_fanout: int,
    sep: str = "\n\n",
) -> GreedyLmNavResult:
    paths = list(iter_root_leaf_paths(root))
    if target_leaf_index < 0 or target_leaf_index >= len(paths):
        raise ValueError(f"target_leaf_index {target_leaf_index} out of range [0,{len(paths)})")
    target_path = paths[target_leaf_index]

    lm.eval()
    head.eval()
    cur = root
    walk: List[TreeNode] = [cur]
    steps: List[GreedyLmNavStep] = []
    goal_t = torch.tensor([target_leaf_index], device=device, dtype=torch.long)

    while not cur.is_leaf():
        depth = len(walk) - 1
        on_gold_prefix = cur is target_path[depth]
        gold = gold_child_index(target_path, cur) if on_gold_prefix else -1

        prefix_doc = path_to_document(walk, sep=sep)
        h = _last_token_hidden(lm, tokenizer, prefix_doc, device, max_length=max_length)
        logits = head(h.unsqueeze(0), goal_t)[0]
        n_ch = len(cur.children)
        scores = logits[:n_ch].float()
        chosen = int(torch.argmax(scores).item())

        losses_per_child = [float(logits[i].cpu()) for i in range(n_ch)]
        correct = on_gold_prefix and (chosen == gold)
        steps.append(
            GreedyLmNavStep(
                depth=depth,
                losses_per_child=losses_per_child,
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


def max_fanout_of_tree(root: TreeNode) -> int:
    m = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if n.children:
            m = max(m, len(n.children))
            stack.extend(n.children)
    return max(m, 1)
