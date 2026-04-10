"""Tests for ``aggregate_ssgs_vs_kv_wikitext_json`` (no torch)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "research" / "aggregate_ssgs_vs_kv_wikitext_json.py"


def _sample_m1_json(*, with_trunc: bool = True, with_l3: bool = False, with_l3_downstream: bool = False) -> dict:
    d: dict = {
        "kind": "ssgs_vs_kv_tree_nav_wikitext",
        "git_sha": "abc",
        "device": "cuda",
        "num_leaves": 8,
        "fanout": 2,
        "chunk_len": 8,
        "dim": 128,
        "target_leaf_index": 7,
        "mamba_layers": 2,
        "tf_layers": 2,
        "tf_nhead": 8,
        "ff_mult": 4,
        "wikitext_config": "wikitext-2-raw-v1",
        "chars_per_leaf": 600,
        "mamba_arm": {
            "ok": True,
            "wall_s": 0.5,
            "peak_alloc_mib": 130.0,
            "snapshots_taken": 7,
            "rollbacks": 11,
            "leaf_checks": 8,
        },
        "tf_kv_clone_arm": {
            "ok": True,
            "wall_s": 0.08,
            "peak_alloc_mib": 27.7,
            "kv_nbytes_at_end": 65536,
        },
    }
    if with_trunc:
        d["tf_kv_truncate_arm"] = {
            "ok": True,
            "wall_s": 0.03,
            "peak_alloc_mib": 27.6,
            "kv_nbytes_at_end": 65536,
            "truncate_kv_calls": 14,
        }
    if with_l3:
        d["l3_tf_kv_hidden"] = {
            "clone_arm": {"cosine_last_token_hidden": 1.0},
            "truncate_arm": {"cosine_last_token_hidden": 1.0000001},
        }
    if with_l3_downstream:
        d["l3_tf_kv_downstream_ce"] = {
            "clone_arm": {"ce_nav": 1.1, "ce_ref": 1.1, "abs_ce_delta": 0.0},
            "truncate_arm": {"ce_nav": 2.0, "ce_ref": 2.0, "abs_ce_delta": 1e-12},
        }
    return d


class TestAggregateSsgsVsKvWikitextJson(unittest.TestCase):
    def test_writes_csv_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            j = tdir / "m1.json"
            j.write_text(json.dumps(_sample_m1_json()), encoding="utf-8")
            out = tdir / "grid.csv"
            r = subprocess.run(
                [sys.executable, str(_SCRIPT), str(j), "--out-csv", str(out)],
                cwd=str(_REPO),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            with out.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["num_leaves"], "8")
            self.assertEqual(rows[0]["tf_kv_truncate_kv_calls"], "14")
            self.assertEqual(rows[0]["l3_clone_cosine"], "")

    def test_l3_columns_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            j = tdir / "m1l3.json"
            j.write_text(json.dumps(_sample_m1_json(with_l3=True)), encoding="utf-8")
            out = tdir / "g.csv"
            subprocess.run(
                [sys.executable, str(_SCRIPT), str(j), "--out-csv", str(out)],
                cwd=str(_REPO),
                check=True,
            )
            with out.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["l3_clone_cosine"], "1.0")
            self.assertIn("1.0000001", rows[0]["l3_truncate_cosine"])

    def test_l3_downstream_ce_columns_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            j = tdir / "m1l3ce.json"
            j.write_text(json.dumps(_sample_m1_json(with_l3_downstream=True)), encoding="utf-8")
            out = tdir / "g.csv"
            subprocess.run(
                [sys.executable, str(_SCRIPT), str(j), "--out-csv", str(out)],
                cwd=str(_REPO),
                check=True,
            )
            with out.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["l3_clone_ce_abs_delta"], "0.0")
            self.assertEqual(rows[0]["l3_truncate_ce_nav"], "2.0")


if __name__ == "__main__":
    unittest.main()
