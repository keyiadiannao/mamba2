"""Pure geometry for leaf-pair cohort labels (A2-S3 task); no torch."""

from __future__ import annotations

import math


def depth_edges(num_leaves: int, fanout: int) -> int:
    d = math.log(num_leaves) / math.log(fanout)
    if abs(d - round(d)) > 1e-9:
        raise ValueError(f"num_leaves={num_leaves} must be fanout**integer_depth for fanout={fanout}")
    return int(round(d))


def block_size(
    cohort: str,
    fanout: int,
    depth: int,
    *,
    custom: int | None = None,
) -> int:
    if cohort == "custom":
        if custom is None or custom < 1:
            raise ValueError("custom block_size required when cohort=custom")
        return int(custom)
    if cohort == "root_child":
        return fanout ** (depth - 1)
    if cohort == "sibling":
        if depth < 2:
            raise ValueError("sibling cohort needs depth>=2")
        return fanout
    raise ValueError(cohort)


def pair_same_cohort_label(i: int, j: int, block: int) -> int:
    return 1 if (i // block) == (j // block) else 0


def all_unordered_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pairs_within_leaf_range(lo: int, hi: int) -> list[tuple[int, int]]:
    """Unordered pairs (i, j) with lo <= i < j < hi."""
    return [(i, j) for i in range(lo, hi) for j in range(i + 1, hi)]
