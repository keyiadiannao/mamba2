#!/usr/bin/env python3
"""
Summarize multiple ``task_wikitext_path_pair`` JSONs (e.g. ``--init-seed`` sweep).

Example::

  python scripts/research/aggregate_task_wikitext_path_pair_json.py \\
    results/metrics/task_wikitext_sibling16_c8_leafheldout6_initseed*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path


def _glob_paths(pattern: str) -> list[Path]:
    recursive = "**" in pattern
    return [Path(s) for s in glob.glob(pattern, recursive=recursive)]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON files (shell-expand globs before calling, or pass via --glob)",
    )
    p.add_argument(
        "--glob",
        "-g",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob under cwd (repeatable); merged with positional paths",
    )
    args = p.parse_args()
    files: list[Path] = []
    for pat in args.glob:
        files.extend(_glob_paths(pat))
    files.extend(args.paths)
    files = sorted({f.resolve() for f in files if f.is_file()})
    if not files:
        print("ERROR: no JSON files matched.", file=sys.stderr)
        return 1

    rows: list[tuple[str, list[float]]] = []
    by_reader: dict[str, list[float]] = {}

    for fp in files:
        data = _load(fp)
        if data.get("kind") != "task_wikitext_path_pair":
            print(f"WARN: skip non-task JSON: {fp}", file=sys.stderr)
            continue
        rc = data.get("ridge_concat") or {}
        if not isinstance(rc, dict):
            continue
        for name, block in rc.items():
            if not isinstance(block, dict):
                continue
            ta = block.get("test_acc")
            if ta is None:
                continue
            by_reader.setdefault(name, []).append(float(ta))

    if not by_reader:
        print("ERROR: no ridge_concat.test_acc found.", file=sys.stderr)
        return 1

    print(f"# files: {len(files)}")
    for name in sorted(by_reader.keys()):
        xs = by_reader[name]
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"{name}\tmean_test_acc={m:.4f}\tstd={s:.4f}\tn={len(xs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
