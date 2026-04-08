"""``tree_lm_closure``：路径文档拼接（无 HF 下载）。"""

from __future__ import annotations

import unittest

import torch

from src.rag_tree.tree import TreeNode
from src.rag_tree.tree_lm_closure import iter_path_documents, path_to_document


class TestPathToDocument(unittest.TestCase):
    def test_skips_empty_text_nodes(self) -> None:
        a = TreeNode(embedding=torch.zeros(2, 4), children=[], text="  hello  ")
        b = TreeNode(embedding=torch.zeros(2, 4), children=[], text="")
        c = TreeNode(embedding=torch.zeros(2, 4), children=[], text="world")
        self.assertEqual(path_to_document([a, b, c], sep="\n"), "hello\nworld")

    def test_iter_path_documents_two_leaves(self) -> None:
        l0 = TreeNode(embedding=torch.zeros(1, 1), children=[], text="L0")
        l1 = TreeNode(embedding=torch.zeros(1, 1), children=[], text="L1")
        root = TreeNode(embedding=torch.zeros(1, 1), children=[l0, l1], text="R")
        pairs = iter_path_documents(root)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], "R\n\nL0")
        self.assertEqual(pairs[1][1], "R\n\nL1")


if __name__ == "__main__":
    unittest.main()
