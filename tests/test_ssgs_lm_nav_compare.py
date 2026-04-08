"""``demo_ssgs_lm_nav_compare`` 所用文本 8 叶树：SSGS 对每个叶目标必达（仅 Mamba，无 LM）。"""

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


_LEAVES8 = [
    "RAG combines retrieval with generation.",
    "Tree RAG uses hierarchical summaries.",
    "Mamba uses a recurrent state for long sequences.",
    "Transformers use quadratic attention in length.",
    "状态空间模型压缩历史为固定维度。",
    "检索头预测是否需要查外部知识。",
    "实验在5060与AutoDL上分工运行。",
    "复现依赖Git与环境锁定文件。",
]


@unittest.skipUnless(_mamba2_available(), "transformers Mamba2Model not available")
class TestSsgsTextTreeEightLeaves(unittest.TestCase):
    def test_dfs_ssgs_mamba_reaches_each_leaf_goal(self) -> None:
        from src.rag_tree.from_text import build_bottom_up_text_tree
        from src.rag_tree.ssgs import (
            MambaNavState,
            SSGSTrace,
            build_toy_mamba2_for_ssgs,
            clear_tree_snapshots,
            dfs_ssgs_mamba,
            mount_mamba_cache_meta_on_tree,
        )
        from src.rag_tree.tree import iter_root_leaf_paths

        dev = torch.device("cpu")
        root = build_bottom_up_text_tree(_LEAVES8, 2, 8, 128, dev, torch.float32)
        leaves = [p[-1] for p in iter_root_leaf_paths(root)]
        self.assertEqual(len(leaves), 8)
        model = build_toy_mamba2_for_ssgs(128, dev, num_layers=2)

        for k, target in enumerate(leaves):
            clear_tree_snapshots(root)
            state = MambaNavState(model=model)
            tr = SSGSTrace()

            def _goal(n: object, t=target) -> bool:
                return n is t

            ok = dfs_ssgs_mamba(
                root,
                state,
                leaf_goal=_goal,
                trace=tr,
                mount_snapshot=mount_mamba_cache_meta_on_tree,
            )
            self.assertTrue(ok, msg=f"goal leaf index {k}")
            self.assertGreater(tr.leaf_checks, 0)


if __name__ == "__main__":
    unittest.main()
