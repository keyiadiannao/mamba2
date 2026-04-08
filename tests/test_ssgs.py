"""SSGS 草稿：假状态 DFS + 快照语义（无 LM 前向）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag_tree.ssgs import SSGSTrace, clear_tree_snapshots, dfs_ssgs, mount_snapshot_on_tree
from src.rag_tree.tree import TreeNode, build_balanced_tree, iter_root_leaf_paths


class TestSSGSDryRun(unittest.TestCase):
    def test_find_rightmost_leaf_binary_depth2(self) -> None:
        # depth=2 fanout=2 => 4 leaves
        root = build_balanced_tree(2, 2, chunk_len=2, dim=4, device=torch.device("cpu"))
        leaves = [path[-1] for path in iter_root_leaf_paths(root)]
        target = leaves[-1]

        def leaf_goal(n: TreeNode) -> bool:
            return n is target

        tr = SSGSTrace()
        ok, st = dfs_ssgs(root, [], leaf_goal=leaf_goal, trace=tr)
        self.assertTrue(ok)
        self.assertEqual(st[-1], id(target))
        self.assertEqual(tr.snapshots_taken, 3)
        self.assertGreater(tr.rollbacks, 0)
        self.assertEqual(tr.leaf_checks, 4)

    def test_mount_snapshot_clears(self) -> None:
        root = build_balanced_tree(1, 2, chunk_len=1, dim=2, device=torch.device("cpu"))
        tr = SSGSTrace()

        def _never(_: TreeNode) -> bool:
            return False

        dfs_ssgs(root, [], leaf_goal=_never, trace=tr, mount_snapshot=mount_snapshot_on_tree)
        self.assertIsNotNone(root.state_snapshot)
        clear_tree_snapshots(root)
        self.assertIsNone(root.state_snapshot)


if __name__ == "__main__":
    unittest.main()
