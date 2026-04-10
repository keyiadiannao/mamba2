"""L3 downstream: fixed leaf-head CE matches nav vs gold-path hidden."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestTfKvL3DownstreamProbe(unittest.TestCase):
    def test_ce_delta_near_zero_balanced_cpu(self) -> None:
        try:
            import torch
        except (OSError, ImportError):
            self.skipTest("PyTorch DLL not loadable on this host (e.g. WinError 1114)")

        from src.rag_tree.tf_kv_l3_downstream_probe import tf_kv_fixed_leaf_head_ce_nav_vs_gold_path
        from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths

        dev = torch.device("cpu")
        g = torch.Generator(device=dev)
        g.manual_seed(42)
        root = build_balanced_tree(2, 2, chunk_len=4, dim=32, device=dev, dtype=torch.float32, generator=g)
        leaves = [path[-1] for path in iter_root_leaf_paths(root)]
        n = len(leaves)
        target = leaves[-1]
        tidx = n - 1

        for trunc in (False, True):
            with self.subTest(truncate=trunc):
                out = tf_kv_fixed_leaf_head_ce_nav_vs_gold_path(
                    root,
                    target,
                    dim=32,
                    tf_layers=2,
                    nhead=4,
                    ff_mult=4,
                    dev=dev,
                    use_truncate_restore=trunc,
                    num_leaves=n,
                    target_leaf_index=tidx,
                    probe_seed=999,
                )
                self.assertTrue(out["dfs_ok"])
                self.assertLess(float(out["abs_ce_delta"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
