"""Load small public text slices from HuggingFace `datasets` for RAPTOR-lite / shallow trees."""

from __future__ import annotations

from typing import List


def wikitext2_leaf_chunks(
    num_leaves: int,
    chars_per_leaf: int = 600,
    *,
    config: str = "wikitext-2-raw-v1",
) -> List[str]:
    """
    Build `num_leaves` contiguous UTF-8 slices from Wikitext-2 (raw).

    Citation: Merity et al., "Pointer Sentinel Mixture Models" (wikitext-103/2).
    HF: `wikitext` loader, config `wikitext-2-raw-v1`.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    if num_leaves < 1 or chars_per_leaf < 1:
        raise ValueError("num_leaves and chars_per_leaf must be positive")

    ds = load_dataset("wikitext", config, split="train")
    parts: list[str] = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if t:
            parts.append(t)
    blob = "\n\n".join(parts)
    need = num_leaves * chars_per_leaf
    if len(blob) < need:
        raise RuntimeError(
            f"wikitext slice too short: got {len(blob)} chars, need {need}. "
            "Lower chars_per_leaf or use wikitext-103-raw-v1 (larger download)."
        )
    blob = blob[:need]
    return [blob[i * chars_per_leaf : (i + 1) * chars_per_leaf] for i in range(num_leaves)]
