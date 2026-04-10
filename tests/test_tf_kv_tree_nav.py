"""TF-KV DFS navigation matches SSGS-Mamba order on a toy balanced tree (CPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestTfKvTreeNav(unittest.TestCase):
    def test_dfs_reaches_rightmost_leaf_balanced_cpu(self) -> None:
        import torch

        from src.rag_tree.ssgs import SSGSTrace, clear_tree_snapshots
        from src.rag_tree.tf_kv_incremental import IncrementalCausalTransformerKV
        from src.rag_tree.tf_kv_tree_nav import TfKvNavState, dfs_tf_kv_nav
        from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths

        dev = torch.device("cpu")
        g = torch.Generator(device=dev)
        g.manual_seed(0)
        root = build_balanced_tree(2, 2, chunk_len=4, dim=32, device=dev, dtype=torch.float32, generator=g)
        leaves_ordered = [path[-1] for path in iter_root_leaf_paths(root)]
        target = leaves_ordered[-1]

        def leaf_goal(node: object) -> bool:
            return node is target

        model = IncrementalCausalTransformerKV(dim=32, nhead=4, num_layers=2, ff_mult=4).to(dev)
        model.eval()
        state = TfKvNavState(model=model)
        tr = SSGSTrace()
        ok = dfs_tf_kv_nav(root, state, leaf_goal=leaf_goal, trace=tr)
        self.assertTrue(ok)
        self.assertEqual(tr.snapshots_taken, 3)
        self.assertEqual(tr.rollbacks, 4)
        self.assertEqual(tr.leaf_checks, 4)
        clear_tree_snapshots(root)

    def test_truncate_matches_clone_counters_balanced_cpu(self) -> None:
        import torch

        from src.rag_tree.ssgs import SSGSTrace, clear_tree_snapshots
        from src.rag_tree.tf_kv_incremental import IncrementalCausalTransformerKV
        from src.rag_tree.tf_kv_tree_nav import TfKvNavState, TfKvTruncateNavState, dfs_tf_kv_nav
        from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths

        dev = torch.device("cpu")
        g = torch.Generator(device=dev)
        g.manual_seed(1)
        root = build_balanced_tree(2, 2, chunk_len=4, dim=32, device=dev, dtype=torch.float32, generator=g)
        leaves_ordered = [path[-1] for path in iter_root_leaf_paths(root)]
        target = leaves_ordered[-1]

        def leaf_goal(node: object) -> bool:
            return node is target

        tr_c = SSGSTrace()
        m_c = IncrementalCausalTransformerKV(dim=32, nhead=4, num_layers=2, ff_mult=4).to(dev)
        m_c.eval()
        self.assertTrue(dfs_tf_kv_nav(root, TfKvNavState(model=m_c), leaf_goal=leaf_goal, trace=tr_c))
        clear_tree_snapshots(root)

        tr_t = SSGSTrace()
        m_t = IncrementalCausalTransformerKV(dim=32, nhead=4, num_layers=2, ff_mult=4).to(dev)
        m_t.eval()
        st = TfKvTruncateNavState(model=m_t)
        self.assertTrue(dfs_tf_kv_nav(root, st, leaf_goal=leaf_goal, trace=tr_t))
        self.assertEqual(tr_c.snapshots_taken, tr_t.snapshots_taken)
        self.assertEqual(tr_c.rollbacks, tr_t.rollbacks)
        self.assertEqual(tr_c.leaf_checks, tr_t.leaf_checks)
        self.assertGreater(st.truncate_kv_calls, 0)
        clear_tree_snapshots(root)

    def test_mamba_and_tf_kv_same_counters_wikitext_shaped_cpu(self) -> None:
        try:
            from transformers import Mamba2Model  # noqa: F401
        except Exception:
            self.skipTest("transformers Mamba2 not available")

        import torch

        from src.rag_tree.from_text import build_bottom_up_text_tree
        from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
        from src.rag_tree.ssgs import (
            MambaNavState,
            SSGSTrace,
            build_toy_mamba2_for_ssgs,
            clear_tree_snapshots,
            dfs_ssgs_mamba,
        )
        from src.rag_tree.tf_kv_incremental import IncrementalCausalTransformerKV
        from src.rag_tree.tf_kv_tree_nav import TfKvNavState, TfKvTruncateNavState, dfs_tf_kv_nav
        from src.rag_tree.tree import iter_root_leaf_paths

        n = 8
        fanout = 2
        chunk_len = 4
        dim = 64
        dev = torch.device("cpu")
        leaves = wikitext2_leaf_chunks(n, 400, config="wikitext-2-raw-v1")
        root = build_bottom_up_text_tree(leaves, fanout, chunk_len, dim, dev, torch.float32)
        leaves_ordered = [path[-1] for path in iter_root_leaf_paths(root)]
        target = leaves_ordered[-1]

        def leaf_goal(node: object) -> bool:
            return node is target

        tr_m = SSGSTrace()
        m_model = build_toy_mamba2_for_ssgs(dim, dev, num_layers=2)
        ok_m = dfs_ssgs_mamba(
            root, MambaNavState(model=m_model), leaf_goal=leaf_goal, trace=tr_m, mount_snapshot=None
        )
        clear_tree_snapshots(root)

        tr_k = SSGSTrace()
        kv_model = IncrementalCausalTransformerKV(dim=dim, nhead=8, num_layers=2, ff_mult=4).to(dev)
        kv_model.eval()
        ok_k = dfs_tf_kv_nav(root, TfKvNavState(model=kv_model), leaf_goal=leaf_goal, trace=tr_k)
        clear_tree_snapshots(root)

        self.assertTrue(ok_m and ok_k)
        self.assertEqual(tr_m.snapshots_taken, tr_k.snapshots_taken)
        self.assertEqual(tr_m.rollbacks, tr_k.rollbacks)
        self.assertEqual(tr_m.leaf_checks, tr_k.leaf_checks)

        clear_tree_snapshots(root)
        tr_t = SSGSTrace()
        kv_t = IncrementalCausalTransformerKV(dim=dim, nhead=8, num_layers=2, ff_mult=4).to(dev)
        kv_t.eval()
        ok_t = dfs_tf_kv_nav(root, TfKvTruncateNavState(model=kv_t), leaf_goal=leaf_goal, trace=tr_t)
        clear_tree_snapshots(root)
        self.assertTrue(ok_t)
        self.assertEqual(tr_m.snapshots_taken, tr_t.snapshots_taken)
        self.assertEqual(tr_m.rollbacks, tr_t.rollbacks)
        self.assertEqual(tr_m.leaf_checks, tr_t.leaf_checks)


if __name__ == "__main__":
    unittest.main()
