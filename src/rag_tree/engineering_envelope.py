"""JSON envelope for `scripts/engineering/*` runs (no torch import)."""


def wrap_path_batch_smoke_envelope(payload: dict, *, runner_relpath: str) -> dict:
    """Wrap `benchmark_wikitext_tree` dict with engineering metadata."""
    return {
        "kind": "engineering_path_batch_smoke",
        "schema_version": 1,
        "runner": runner_relpath,
        "underlying_kind": payload.get("kind"),
        "arms_note": (
            "Sprint-1: single path-batch bundle (TF+GRU+Mamba2 readers per benchmark_wikitext_tree). "
            "Later: split explicit Mamba vs HF-AutoModelForCausalLM arms under §Ⅷ-0."
        ),
        "payload": payload,
    }
