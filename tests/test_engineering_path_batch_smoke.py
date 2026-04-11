"""Engineering envelope schema (no torch)."""
from __future__ import annotations

from src.rag_tree.engineering_envelope import wrap_path_batch_smoke_envelope


def test_wrap_path_batch_smoke_envelope_schema() -> None:
    payload = {"kind": "benchmark_wikitext_tree", "git_sha": "abc1234"}
    out = wrap_path_batch_smoke_envelope(
        payload,
        runner_relpath="scripts/engineering/run_engineering_path_batch_smoke.py",
    )
    assert out["kind"] == "engineering_path_batch_smoke"
    assert out["schema_version"] == 1
    assert out["payload"] == payload
    assert out["underlying_kind"] == "benchmark_wikitext_tree"
