"""CLI smoke for G3-b (no torch / no Hub download)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_g3_compare_help_exits_zero() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "engineering" / "run_g3_causal_lm_compare.py"
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "engineering_causal_lm_compare" in (r.stdout + r.stderr)
