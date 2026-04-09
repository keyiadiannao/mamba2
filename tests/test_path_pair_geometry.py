"""Unit tests for leaf-pair cohort geometry (A2-S3)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag_tree.path_pair_geometry import (
    all_unordered_pairs,
    block_size,
    depth_edges,
    pair_same_cohort_label,
    pairs_within_leaf_range,
)


class TestPathPairGeometry(unittest.TestCase):
    def test_depth_edges_16_fanout2(self) -> None:
        self.assertEqual(depth_edges(16, 2), 4)

    def test_block_sibling_vs_root_child_16(self) -> None:
        d = depth_edges(16, 2)
        self.assertEqual(block_size("sibling", 2, d), 2)
        self.assertEqual(block_size("root_child", 2, d), 8)

    def test_pairs_within_count(self) -> None:
        self.assertEqual(len(pairs_within_leaf_range(0, 10)), 45)
        self.assertEqual(len(pairs_within_leaf_range(10, 16)), 15)

    def test_pair_label_sibling_block2(self) -> None:
        b = 2
        self.assertEqual(pair_same_cohort_label(0, 1, b), 1)
        self.assertEqual(pair_same_cohort_label(0, 2, b), 0)

    def test_all_unordered_pairs_n4(self) -> None:
        p = all_unordered_pairs(4)
        self.assertEqual(len(p), 6)
        self.assertIn((0, 3), p)


if __name__ == "__main__":
    unittest.main()
