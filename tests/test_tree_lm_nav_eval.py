"""``tree_lm_nav_eval``：金子下标与贪心步（不依赖 HF 下载）。"""

from __future__ import annotations

import unittest

import torch

from src.rag_tree.from_text import build_bottom_up_text_tree
from src.rag_tree.tree import iter_root_leaf_paths
from src.rag_tree.tree_lm_nav_eval import gold_child_index, greedy_navigate_by_lm_child_loss


class TestGoldChildIndex(unittest.TestCase):
    def test_depth2_fanout2(self) -> None:
        dev = torch.device("cpu")
        lines = ["a", "b", "c", "d"]
        root = build_bottom_up_text_tree(lines, 2, 2, 16, dev, torch.float32)
        paths = list(iter_root_leaf_paths(root))
        self.assertEqual(len(paths), 4)
        # 左先 DFS：0=a,1=b,2=c,3=d
        p3 = paths[3]
        self.assertEqual(gold_child_index(p3, root), 1)
        self.assertEqual(gold_child_index(p3, root.children[1]), 1)
        p0 = paths[0]
        self.assertEqual(gold_child_index(p0, root), 0)
        self.assertEqual(gold_child_index(p0, root.children[0]), 0)


class TestGreedyWithMockModel(unittest.TestCase):
    def test_mock_always_first_child_reaches_leftmost(self) -> None:
        """恒定 loss → argmin 取 0；目标叶 0 时应到达。"""

        class _Tok:
            def __call__(self, text: str, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

        class _Out:
            loss = torch.tensor(0.5)

        class _Model(torch.nn.Module):
            def forward(self, **kwargs):
                return _Out()

        dev = torch.device("cpu")
        lines = ["a", "b", "c", "d"]
        root = build_bottom_up_text_tree(lines, 2, 2, 8, dev, torch.float32)
        m = _Model()
        tok = _Tok()
        r = greedy_navigate_by_lm_child_loss(root, 0, m, tok, dev, max_length=64)
        self.assertTrue(r.reached_target_leaf)
        self.assertEqual(r.child_choice_accuracy, 1.0)

        r_wrong = greedy_navigate_by_lm_child_loss(root, 3, m, tok, dev, max_length=64)
        self.assertFalse(r_wrong.reached_target_leaf)


if __name__ == "__main__":
    unittest.main()
