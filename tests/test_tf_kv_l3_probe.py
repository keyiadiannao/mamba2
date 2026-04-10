"""TF-KV L3: DFS+restore last hidden matches gold-path-only forward."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestTfKvL3Probe(unittest.TestCase):
    def test_clone_and_truncate_cosine_near_one_balanced_cpu(self) -> None:
        try:
            import torch
        except OSError:
            self.skipTest("PyTorch DLL not loadable on this host (e.g. WinError 1114)")

        from src.rag_tree.tf_kv_l3_probe import tf_kv_hidden_consistency_nav_vs_gold_path
        from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths

        dev = torch.device("cpu")
        g = torch.Generator(device=dev)
        g.manual_seed(42)
        root = build_balanced_tree(2, 2, chunk_len=4, dim=32, device=dev, dtype=torch.float32, generator=g)
        leaves = [path[-1] for path in iter_root_leaf_paths(root)]
        target = leaves[-1]

        for trunc in (False, True):
            with self.subTest(truncate=trunc):
                out = tf_kv_hidden_consistency_nav_vs_gold_path(
                    root,
                    target,
                    dim=32,
                    tf_layers=2,
                    nhead=4,
                    ff_mult=4,
                    dev=dev,
                    use_truncate_restore=trunc,
                )
                self.assertTrue(out["dfs_ok"])
                self.assertGreater(float(out["cosine_last_token_hidden"]), 0.9999)
                self.assertLess(float(out["l2_diff_last_token_hidden"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
