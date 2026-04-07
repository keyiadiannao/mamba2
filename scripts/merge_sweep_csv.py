#!/usr/bin/env python3
"""
Merge multiple sweep CSVs into one file (union of columns). Missing cells become empty strings.

  python scripts/merge_sweep_csv.py results/metrics/merged.csv results/metrics/a.csv results/metrics/b.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out_csv", type=str)
    p.add_argument("inputs", nargs="+", type=str)
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []

    for path_str in args.inputs:
        path = Path(path_str)
        if not path.is_file():
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for c in cols:
                if c not in fieldnames:
                    fieldnames.append(c)
            for row in reader:
                r = {c: (row.get(c) or "") for c in cols}
                rows.append(r)
                for c in r:
                    if c not in fieldnames:
                        fieldnames.append(c)

    for r in rows:
        for c in fieldnames:
            r.setdefault(c, "")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fieldnames})

    print(f"merged {len(rows)} rows -> {out} (cols={len(fieldnames)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
