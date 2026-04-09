"""Tree-structured RAG (e.g. RAPTOR-style): build, traverse, evaluate."""

from __future__ import annotations

from typing import Any

__all__ = [
    "TreeNode",
    "build_balanced_tree",
    "iter_root_leaf_paths",
    "path_tensor",
    "batched_paths",
]


def __getattr__(name: str) -> Any:
    """Lazy re-exports so ``import src.rag_tree.path_pair_geometry`` does not load torch."""
    if name in __all__:
        from src.rag_tree import tree as _tree

        return getattr(_tree, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
