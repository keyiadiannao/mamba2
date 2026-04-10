"""Tests for trajectory A vs B L3 probe (no full-tree DFS)."""

from __future__ import annotations

import unittest

import torch

from src.rag_tree.tf_kv_trajectory_l3 import compare_trajectories_ab


class TestTfKvTrajectoryL3(unittest.TestCase):
    def test_cosine_near_one_cpu_clone_and_truncate(self) -> None:
        dev = torch.device("cpu")
        for use_tr in (False, True):
            payload, res = compare_trajectories_ab(
                depth=2,
                fanout=2,
                chunk_len=4,
                dim=32,
                tf_layers=2,
                nhead=4,
                ff_mult=2,
                device=dev,
                init_seed=0,
                use_truncate_restore=use_tr,
            )
            self.assertGreater(res.cosine_ab, 0.9999, msg=payload.get("arm"))
            self.assertLess(res.l2_ab, 5e-4)  # float32; Linux CI / CUDA 可更紧
            self.assertEqual(res.gold_path_nodes, 3)
            self.assertFalse(res.wrong_branch_node_is_leaf)


if __name__ == "__main__":
    unittest.main()
