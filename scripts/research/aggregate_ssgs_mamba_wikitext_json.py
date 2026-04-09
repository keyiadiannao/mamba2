#!/usr/bin/env python3
"""
Merge one or more ``ssgs_mamba_wikitext_tree`` JSONs (from ``demo_ssgs_mamba_wikitext.py``)
into a single CSV for archiving next to path-batch grids.

Examples::

  python scripts/research/aggregate_ssgs_mamba_wikitext_json.py \\
    -g 'results/metrics_result/ssgs_mamba_wikitext_*.json' \\
    --out-csv results/metrics_result/ssgs_mamba_wikitext_grid.csv

  # Merge with an existing CSV (union by absolute ``json_path``; later files win on key clash)
  python scripts/research/aggregate_ssgs_mamba_wikitext_json.py --append \\
    -g '$MAMBA2_RESULTS_ROOT/metrics_result/ssgs_mamba_wikitext_*.json' \\
    --out-csv results/metrics_result/ssgs_mamba_wikitext_grid.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


_KIND = "ssgs_mamba_wikitext_tree"

_CSV_FIELDS = [
    "json_path",
    "git_sha",
    "device",
    "num_leaves",
    "fanout",
    "chunk_len",
    "dim",
    "mamba_layers",
    "target_leaf_index",
    "ok",
    "snapshots_taken",
    "rollbacks",
    "leaf_checks",
    "wikitext_config",
    "chars_per_leaf",
    "mamba_torch_forward_only",
    "kind",
]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _row_from_json(abs_path: Path, data: dict) -> dict[str, object]:
    p = str(abs_path.resolve())
    return {
        "json_path": p,
        "git_sha": data.get("git_sha", ""),
        "device": data.get("device", ""),
        "num_leaves": data.get("num_leaves", ""),
        "fanout": data.get("fanout", ""),
        "chunk_len": data.get("chunk_len", ""),
        "dim": data.get("dim", ""),
        "mamba_layers": data.get("mamba_layers", ""),
        "target_leaf_index": data.get("target_leaf_index", ""),
        "ok": data.get("ok", ""),
        "snapshots_taken": data.get("snapshots_taken", ""),
        "rollbacks": data.get("rollbacks", ""),
        "leaf_checks": data.get("leaf_checks", ""),
        "wikitext_config": data.get("wikitext_config", ""),
        "chars_per_leaf": data.get("chars_per_leaf", ""),
        "mamba_torch_forward_only": data.get("mamba_torch_forward_only", ""),
        "kind": data.get("kind", _KIND),
    }


def _glob_paths(pattern: str) -> list[Path]:
    """Expand a glob; supports absolute paths (``pathlib.Path.glob`` does not on Py3.11+)."""
    recursive = "**" in pattern
    return [Path(s) for s in glob.glob(pattern, recursive=recursive)]


def _read_existing_csv(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = row.get("json_path") or ""
            if key:
                out[key] = {k: row.get(k, "") for k in _CSV_FIELDS}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON files (expand globs in shell, or use --glob)",
    )
    p.add_argument(
        "--glob",
        "-g",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob pattern (repeatable); relative to cwd or absolute, e.g. $ROOT/metrics_result/*.json",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Load existing CSV (if any) and union rows by json_path; default writes only rows from matched JSON files",
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

    by_path: dict[str, dict[str, object]] = {}
    if args.append:
        by_path = _read_existing_csv(args.out_csv)

    n_ok = 0
    for fp in files:
        data = _load(fp)
        if data.get("kind") != _KIND:
            print(f"WARN: skip (kind != {_KIND}): {fp}", file=sys.stderr)
            continue
        row = _row_from_json(fp, data)
        by_path[str(row["json_path"])] = row
        n_ok += 1

    if n_ok == 0 and not by_path:
        print("ERROR: no matching JSON rows.", file=sys.stderr)
        return 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [by_path[k] for k in sorted(by_path.keys())]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out_csv} ({len(rows)} row(s)); merged from inputs this run: {n_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
