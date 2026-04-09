"""Tests for ``aggregate_ssgs_mamba_wikitext_json`` (no torch)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "research" / "aggregate_ssgs_mamba_wikitext_json.py"


class TestAggregateSsgsMambaWikitextJson(unittest.TestCase):
    def test_writes_csv_from_json(self) -> None:
        sample = {
            "kind": "ssgs_mamba_wikitext_tree",
            "git_sha": "abc1234",
            "device": "cpu",
            "num_leaves": 8,
            "fanout": 2,
            "chunk_len": 4,
            "dim": 64,
            "mamba_layers": 2,
            "target_leaf_index": 7,
            "ok": True,
            "snapshots_taken": 5,
            "rollbacks": 3,
            "leaf_checks": 4,
            "wikitext_config": "wikitext-2-raw-v1",
            "chars_per_leaf": 400,
            "mamba_torch_forward_only": False,
        }
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            j = tdir / "a.json"
            j.write_text(json.dumps(sample), encoding="utf-8")
            out = tdir / "grid.csv"
            r = subprocess.run(
                [sys.executable, str(_SCRIPT), str(j), "--out-csv", str(out)],
                cwd=str(_REPO),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertTrue(out.is_file())
            with out.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["git_sha"], "abc1234")
            self.assertEqual(rows[0]["snapshots_taken"], "5")
            self.assertIn("a.json", rows[0]["json_path"])

    def test_absolute_glob_pattern(self) -> None:
        """Py3.11+ pathlib rejects absolute globs; script must use stdlib glob."""
        sample = {
            "kind": "ssgs_mamba_wikitext_tree",
            "git_sha": "x",
            "device": "cpu",
            "num_leaves": 8,
            "fanout": 2,
            "chunk_len": 4,
            "dim": 64,
            "mamba_layers": 2,
            "target_leaf_index": 0,
            "ok": True,
            "snapshots_taken": 1,
            "rollbacks": 0,
            "leaf_checks": 1,
            "wikitext_config": "wikitext-2-raw-v1",
            "chars_per_leaf": 400,
            "mamba_torch_forward_only": False,
        }
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            j = tdir / "b.json"
            j.write_text(json.dumps(sample), encoding="utf-8")
            out = tdir / "grid.csv"
            pat = str(tdir / "*.json")
            r = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "-g",
                    pat,
                    "--out-csv",
                    str(out),
                ],
                cwd=str(_REPO),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            with out.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
