"""`run_engineering.py` 无 GPU：仅校验帮助与未知子命令。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_engineering_help_exits_zero() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "engineering" / "run_engineering.py"
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert "path-batch-smoke" in out
    assert "m1-ssgs-vs-kv" in out


def test_run_engineering_unknown_command() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "engineering" / "run_engineering.py"
    r = subprocess.run(
        [sys.executable, str(script), "not-a-real-subcommand"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
