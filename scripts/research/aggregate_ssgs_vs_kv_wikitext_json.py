#!/usr/bin/env python3
"""
Merge **M1** JSONs (**``kind=ssgs_vs_kv_tree_nav_wikitext``**, from
``benchmark_ssgs_vs_kv_tree_nav_wikitext.py``) into one CSV (alongside path-batch / SSGS-Mamba grids).

Examples::

  python scripts/research/aggregate_ssgs_vs_kv_wikitext_json.py \\
    -g 'results/metrics_result/ssgs_vs_kv_tree_nav_wikitext_*.json' \\
    --out-csv results/metrics_result/ssgs_vs_kv_wikitext_nav_grid.csv

  AGGREGATE_APPEND=1 bash scripts/server/run_m1_ssgs_vs_kv_wikitext_cuda.sh
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

_KIND = "ssgs_vs_kv_tree_nav_wikitext"

_CSV_FIELDS = [
    "json_path",
    "git_sha",
    "device",
    "num_leaves",
    "fanout",
    "chunk_len",
    "dim",
    "target_leaf_index",
    "mamba_layers",
    "tf_layers",
    "tf_nhead",
    "ff_mult",
    "wikitext_config",
    "chars_per_leaf",
    "mamba_ok",
    "mamba_wall_s",
    "mamba_peak_alloc_mib",
    "mamba_snapshots_taken",
    "mamba_rollbacks",
    "mamba_leaf_checks",
    "tf_kv_clone_ok",
    "tf_kv_clone_wall_s",
    "tf_kv_clone_peak_alloc_mib",
    "tf_kv_clone_kv_nbytes_at_end",
    "tf_kv_truncate_ok",
    "tf_kv_truncate_wall_s",
    "tf_kv_truncate_peak_alloc_mib",
    "tf_kv_truncate_kv_nbytes_at_end",
    "tf_kv_truncate_kv_calls",
    "l3_clone_cosine",
    "l3_truncate_cosine",
    "l3_clone_ce_nav",
    "l3_clone_ce_ref",
    "l3_clone_ce_abs_delta",
    "l3_truncate_ce_nav",
    "l3_truncate_ce_ref",
    "l3_truncate_ce_abs_delta",
    "kind",
]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _row_from_json(abs_path: Path, data: dict) -> dict[str, object]:
    ma = data.get("mamba_arm") or {}
    cl = data.get("tf_kv_clone_arm") or data.get("tf_kv_arm") or {}
    tr = data.get("tf_kv_truncate_arm") or {}
    l3 = data.get("l3_tf_kv_hidden") if isinstance(data.get("l3_tf_kv_hidden"), dict) else {}
    l3c = l3.get("clone_arm") if isinstance(l3.get("clone_arm"), dict) else {}
    l3t = l3.get("truncate_arm") if isinstance(l3.get("truncate_arm"), dict) else {}
    l3ce = data.get("l3_tf_kv_downstream_ce") if isinstance(data.get("l3_tf_kv_downstream_ce"), dict) else {}
    l3cec = l3ce.get("clone_arm") if isinstance(l3ce.get("clone_arm"), dict) else {}
    l3cet = l3ce.get("truncate_arm") if isinstance(l3ce.get("truncate_arm"), dict) else {}

    def _f(x: object) -> object:
        return "" if x is None else x

    return {
        "json_path": str(abs_path.resolve()),
        "git_sha": _f(data.get("git_sha")),
        "device": _f(data.get("device")),
        "num_leaves": _f(data.get("num_leaves")),
        "fanout": _f(data.get("fanout")),
        "chunk_len": _f(data.get("chunk_len")),
        "dim": _f(data.get("dim")),
        "target_leaf_index": _f(data.get("target_leaf_index")),
        "mamba_layers": _f(data.get("mamba_layers")),
        "tf_layers": _f(data.get("tf_layers")),
        "tf_nhead": _f(data.get("tf_nhead")),
        "ff_mult": _f(data.get("ff_mult")),
        "wikitext_config": _f(data.get("wikitext_config")),
        "chars_per_leaf": _f(data.get("chars_per_leaf")),
        "mamba_ok": _f(ma.get("ok")),
        "mamba_wall_s": _f(ma.get("wall_s")),
        "mamba_peak_alloc_mib": _f(ma.get("peak_alloc_mib")),
        "mamba_snapshots_taken": _f(ma.get("snapshots_taken")),
        "mamba_rollbacks": _f(ma.get("rollbacks")),
        "mamba_leaf_checks": _f(ma.get("leaf_checks")),
        "tf_kv_clone_ok": _f(cl.get("ok")),
        "tf_kv_clone_wall_s": _f(cl.get("wall_s")),
        "tf_kv_clone_peak_alloc_mib": _f(cl.get("peak_alloc_mib")),
        "tf_kv_clone_kv_nbytes_at_end": _f(cl.get("kv_nbytes_at_end")),
        "tf_kv_truncate_ok": _f(tr.get("ok")) if tr else "",
        "tf_kv_truncate_wall_s": _f(tr.get("wall_s")) if tr else "",
        "tf_kv_truncate_peak_alloc_mib": _f(tr.get("peak_alloc_mib")) if tr else "",
        "tf_kv_truncate_kv_nbytes_at_end": _f(tr.get("kv_nbytes_at_end")) if tr else "",
        "tf_kv_truncate_kv_calls": _f(tr.get("truncate_kv_calls")) if tr else "",
        "l3_clone_cosine": _f(l3c.get("cosine_last_token_hidden")),
        "l3_truncate_cosine": _f(l3t.get("cosine_last_token_hidden")),
        "l3_clone_ce_nav": _f(l3cec.get("ce_nav")),
        "l3_clone_ce_ref": _f(l3cec.get("ce_ref")),
        "l3_clone_ce_abs_delta": _f(l3cec.get("abs_ce_delta")),
        "l3_truncate_ce_nav": _f(l3cet.get("ce_nav")) if l3cet else "",
        "l3_truncate_ce_ref": _f(l3cet.get("ce_ref")) if l3cet else "",
        "l3_truncate_ce_abs_delta": _f(l3cet.get("abs_ce_delta")) if l3cet else "",
        "kind": _f(data.get("kind")) or _KIND,
    }


def _glob_paths(pattern: str) -> list[Path]:
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
    p.add_argument("paths", nargs="*", type=Path, help="JSON files")
    p.add_argument(
        "--glob",
        "-g",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob pattern (repeatable)",
    )
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    p.add_argument(
        "--append",
        action="store_true",
        help="Union with existing CSV by json_path",
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
