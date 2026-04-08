"""SSGS + HF Mamba2Model DynamicCache（需 transformers Mamba2）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _mamba2_available() -> bool:
    try:
        from transformers import Mamba2Model  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_mamba2_available(), "transformers Mamba2Model not available")
class TestSSGSMambaNav(unittest.TestCase):
    def test_dfs_finds_rightmost_leaf_depth2(self) -> None:
        from src.rag_tree.ssgs import (
            MambaNavState,
            SSGSTrace,
            build_toy_mamba2_for_ssgs,
            clear_tree_snapshots,
            dfs_ssgs_mamba,
            mount_mamba_cache_meta_on_tree,
        )
        from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths

        dim = 32
        dev = torch.device("cpu")
        root = build_balanced_tree(2, 2, chunk_len=2, dim=dim, device=dev)
        leaves = [path[-1] for path in iter_root_leaf_paths(root)]
        target = leaves[-1]

        def leaf_goal(n) -> bool:
            return n is target

        model = build_toy_mamba2_for_ssgs(dim, dev, num_layers=2)
        state = MambaNavState(model=model)
        tr = SSGSTrace()
        ok = dfs_ssgs_mamba(
            root,
            state,
            leaf_goal=leaf_goal,
            trace=tr,
            mount_snapshot=mount_mamba_cache_meta_on_tree,
        )
        self.assertTrue(ok)
        self.assertEqual(tr.snapshots_taken, 3)
        self.assertGreater(tr.rollbacks, 0)
        self.assertIsNotNone(root.state_snapshot)
        self.assertEqual(root.state_snapshot.get("kind"), "mamba_dynamic_cache")
        clear_tree_snapshots(root)
        self.assertIsNone(root.state_snapshot)

    def test_restore_roundtrip_matches_snapshot(self) -> None:
        from src.rag_tree.mamba_cache_utils import clone_mamba_dynamic_cache, zero_mamba_dynamic_cache
        from src.rag_tree.ssgs import MambaNavState, build_toy_mamba2_for_ssgs
        from src.rag_tree.tree import build_balanced_tree

        dim = 32
        dev = torch.device("cpu")
        root = build_balanced_tree(0, 2, chunk_len=2, dim=dim, device=dev)
        model = build_toy_mamba2_for_ssgs(dim, dev, num_layers=1)
        state = MambaNavState(model=model)
        state.absorb_node(root)
        assert state._cache is not None
        snap = clone_mamba_dynamic_cache(state._cache)
        zero_mamba_dynamic_cache(state._cache)
        state.restore(snap)
        snap2 = clone_mamba_dynamic_cache(state._cache)
        for a, b in zip(snap, snap2):
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]))


if __name__ == "__main__":
    unittest.main()
