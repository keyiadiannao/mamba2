#!/usr/bin/env python3
"""
Build a UTF-8 leaf file (one passage per line) from a directory of .txt files.

For use with `benchmark_text_tree.py`: line count should equal fanout**depth.

Example:
  python scripts/data/prepare_leaves_from_corpus.py --input-dir data/raw/sample --out results/metrics/leaves_from_repo_sample.txt --fanout 2 --depth 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="data/raw/sample")
    p.add_argument("--glob", type=str, default="*.txt")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--fanout", type=int, default=0, help="If >0, require exactly fanout**depth lines")
    p.add_argument("--depth", type=int, default=0)
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"ERROR: not a directory: {in_dir}", file=sys.stderr)
        return 1

    files = sorted(in_dir.glob(args.glob))
    lines: list[str] = []
    for f in files:
        if not f.is_file() or f.name.startswith("."):
            continue
        raw = f.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        one_line = " ".join(raw.split())
        lines.append(one_line)

    need = 0
    if args.fanout > 0 and args.depth > 0:
        need = args.fanout**args.depth
        if len(lines) != need:
            print(
                f"ERROR: got {len(lines)} non-empty texts, need fanout**depth = {need} "
                f"(fanout={args.fanout}, depth={args.depth})",
                file=sys.stderr,
            )
            return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {len(lines)} lines -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
