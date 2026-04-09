"""SSGS + Mamba on Wikitext-shaped tree (same build as ``benchmark_wikitext_tree``)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _mamba2_available() -> bool:
    try:
        from transformers import Mamba2Model  # noqa: F401

        return True
    except Exception:
        # ImportError / OSError (broken PyTorch DLL); rare hard failures during torch init — never break pytest collection.
        return False


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_mamba2_available() and _datasets_available(), "Mamba2 + datasets required")
class TestSSGSMambaWikitext(unittest.TestCase):
    def test_dfs_reaches_target_leaf_n8_cpu(self) -> None:
        import torch

        from src.rag_tree.from_text import build_bottom_up_text_tree
        from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
        from src.rag_tree.ssgs import (
            MambaNavState,
            SSGSTrace,
            build_toy_mamba2_for_ssgs,
            clear_tree_snapshots,
            dfs_ssgs_mamba,
            mount_mamba_cache_meta_on_tree,
        )
        from src.rag_tree.tree import iter_root_leaf_paths

        n = 8
        fanout = 2
        chunk_len = 4
        dim = 64
        dev = torch.device("cpu")
        leaves = wikitext2_leaf_chunks(n, 400, config="wikitext-2-raw-v1")
        root = build_bottom_up_text_tree(leaves, fanout, chunk_len, dim, dev, torch.float32)
        leaves_ordered = [path[-1] for path in iter_root_leaf_paths(root)]
        self.assertEqual(len(leaves_ordered), n)
        target = leaves_ordered[-1]

        def leaf_goal(node: object) -> bool:
            return node is target

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
        self.assertGreater(tr.snapshots_taken, 0)
        self.assertGreaterEqual(tr.leaf_checks, 1)
        clear_tree_snapshots(root)


if __name__ == "__main__":
    unittest.main()
