"""``tree_lm_nav_learned``：监督样本数量与 max_fanout（无 HF）。"""

from __future__ import annotations

import unittest

import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.tree_lm_nav_learned import (
    iter_goal_conditioned_examples,
    max_fanout_of_tree,
)


class TestGoalConditionedExamples(unittest.TestCase):
    def test_four_leaves_count(self) -> None:
        dev = torch.device("cpu")
        root = build_bottom_up_text_tree(["a", "b", "c", "d"], 2, 2, 16, dev, torch.float32)
        ex = iter_goal_conditioned_examples(root)
        # 4 条叶路径 × 每路径 2 个内部节点 = 8
        self.assertEqual(len(ex), 8)
        self.assertEqual(max_fanout_of_tree(root), 2)


if __name__ == "__main__":
    unittest.main()
