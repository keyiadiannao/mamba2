#!/usr/bin/env python3
"""
G3-b：预训因果 LM，baseline（全路径文本拼接、单次截断前向）vs 树路径逐条前向；kind=engineering_causal_lm_compare。

协议与主表列见 ENGINEERING_NORTH_STAR_PLAN.md §4.3、§4.4。示例命令见 scripts/engineering/README.md。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _git_short_sha(repo: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return "unknown"


def _load_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)


def _tokenizer_hub_id_for_arm(
    arm_key: str, model_id: str, gpt2_id: str, mamba_tokenizer_id: str | None
) -> str:
    if arm_key != "mamba2_370m":
        return model_id
    if mamba_tokenizer_id is not None:
        return mamba_tokenizer_id
    if model_id == "state-spaces/mamba2-370m":
        return gpt2_id
    return model_id


def _forward_peak_cache_time(
    model, enc: dict, device, dtype_el_size: int
) -> dict:
    """单次 ``use_cache=True`` 前向；返回 seq_len、峰值 MiB、cache nbytes、墙钟。"""
    import torch

    from src.rag_tree.causal_lm_kv_stats import (
        cache_nbytes_from_outputs,
        estimate_attention_kv_nbytes_mha_stack,
    )

    enc = {k: v.to(device) for k, v in enc.items()}
    seq_len = int(enc["input_ids"].shape[1])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**enc, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    peak_mib = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        if device.type == "cuda"
        else None
    )
    nbytes = int(cache_nbytes_from_outputs(outputs))
    row: dict = {
        "seq_len": seq_len,
        "peak_alloc_mib": peak_mib,
        "past_or_cache_nbytes": nbytes,
        "elapsed_s": float(t1 - t0),
    }
    if nbytes == 0:
        est = estimate_attention_kv_nbytes_mha_stack(
            model.config, seq_len, batch=1, element_size=dtype_el_size
        )
        if est is not None:
            row["past_or_cache_nbytes_estimated"] = est
    return row


def _run_arm(
    *,
    model_id: str,
    tokenizer,
    model,
    device,
    dtype_el_size: int,
    baseline_text: str,
    path_docs: list[tuple],
    max_length: int,
) -> dict:
    """baseline 单次 forward + 树路径逐路径 forward。"""
    per_path_rows = []
    peak_max = None
    sum_seq = 0
    sum_elapsed = 0.0
    for i, (_path, doc) in enumerate(path_docs):
        r: dict = {"path_index": i, "doc_chars": len(doc)}
        if not (doc or "").strip():
            r["skip"] = "empty_document"
            per_path_rows.append(r)
            continue
        enc = tokenizer(
            doc,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        st = _forward_peak_cache_time(model, enc, device, dtype_el_size)
        r.update(st)
        sl = int(st["seq_len"])
        sum_seq += sl
        sum_elapsed += float(st["elapsed_s"])
        pm = st["peak_alloc_mib"]
        if pm is not None:
            peak_max = pm if peak_max is None else max(peak_max, pm)
        per_path_rows.append(r)

    baseline_row: dict | None = None
    if (baseline_text or "").strip():
        enc_b = tokenizer(
            baseline_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        baseline_row = _forward_peak_cache_time(model, enc_b, device, dtype_el_size)
    else:
        baseline_row = {"skip": "empty_baseline_concat", "seq_len": 0}

    ratios: dict = {}
    b_peak = baseline_row.get("peak_alloc_mib")
    b_seq = baseline_row.get("seq_len")
    if (
        isinstance(b_peak, (int, float))
        and b_peak > 0
        and peak_max is not None
    ):
        ratios["peak_max_tree_over_baseline"] = float(peak_max) / float(b_peak)
    if isinstance(b_seq, int) and b_seq > 0 and sum_seq > 0:
        ratios["token_steps_sum_tree_over_baseline_seq"] = float(sum_seq) / float(
            b_seq
        )

    return {
        "model_id": model_id,
        "device": str(device),
        "protocol": {
            "baseline": "concat_all_path_documents_single_forward_trunc_max_length",
            "tree": "per_path_separate_forward_trunc_max_length",
            "note": "tree peak_alloc uses max over paths; token upper bound: per-path seq_len sum vs baseline seq_len",
        },
        "baseline_ar": baseline_row,
        "tree_path": {
            "num_paths": len(path_docs),
            "sum_seq_lens": sum_seq,
            "max_peak_alloc_mib": peak_max,
            "sum_elapsed_s": sum_elapsed,
            "paths": per_path_rows,
        },
        "within_arm_ratios": ratios,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-paths", type=int, default=0, help="0 = all root-leaf paths")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpt2-id", type=str, default="openai-community/gpt2")
    p.add_argument("--mamba-id", type=str, default="AntonV/mamba2-370m-hf")
    p.add_argument("--mamba-tokenizer-id", type=str, default=None)
    p.add_argument("--mamba", action="store_true")
    p.add_argument("--gpt2-only", action="store_true")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--git-sha", type=str, default=None)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from src.rag_tree.from_text import build_bottom_up_text_tree
    from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
    from src.rag_tree.tree_lm_closure import ensure_causal_lm_tokenizer, iter_path_documents

    try:
        leaves = wikitext2_leaf_chunks(
            args.num_leaves, args.chars_per_leaf, config=args.wikitext_config
        )
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    root = build_bottom_up_text_tree(
        leaves,
        args.fanout,
        args.chunk_len,
        args.dim,
        device,
        torch.float32,
    )
    path_docs_full = iter_path_documents(root, sep="\n\n")
    num_paths_total = len(path_docs_full)
    path_docs = path_docs_full
    if args.max_paths > 0:
        path_docs = path_docs[: args.max_paths]

    baseline_parts = [(d or "").strip() for _p, d in path_docs if (d or "").strip()]
    baseline_text = "\n\n".join(baseline_parts)

    model_ids: list[tuple[str, str]] = [(args.gpt2_id, "gpt2")]
    if args.mamba and not args.gpt2_only:
        model_ids.append((args.mamba_id, "mamba2_370m"))
    if args.gpt2_only:
        model_ids = [(args.gpt2_id, "gpt2")]

    arms: dict = {}
    for model_id, key in model_ids:
        try:
            tok_id = _tokenizer_hub_id_for_arm(
                key, model_id, args.gpt2_id, args.mamba_tokenizer_id
            )
            tokenizer = _load_tokenizer(tok_id)
            ensure_causal_lm_tokenizer(tokenizer)
            dtype = torch.float32 if device.type == "cpu" else torch.float16
            es = 4 if dtype == torch.float32 else 2
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            model.eval()
            model.to(device)
        except Exception as e:
            arms[key] = {"model_id": model_id, "error": str(e)}
            continue

        arm_out = _run_arm(
            model_id=model_id,
            tokenizer=tokenizer,
            model=model,
            device=device,
            dtype_el_size=es,
            baseline_text=baseline_text,
            path_docs=path_docs,
            max_length=args.max_length,
        )
        tok_id = _tokenizer_hub_id_for_arm(
            key, model_id, args.gpt2_id, args.mamba_tokenizer_id
        )
        if tok_id != model_id:
            arm_out["tokenizer_id"] = tok_id
        arm_out["dtype"] = str(dtype)
        arms[key] = arm_out
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out: dict = {
        "kind": "engineering_causal_lm_compare",
        "schema_version": 1,
        "runner": "scripts/engineering/run_g3_causal_lm_compare.py",
        "tree_kind": "wikitext2_bottom_up",
        "dataset": "wikitext",
        "wikitext_config": args.wikitext_config,
        "num_leaves": args.num_leaves,
        "fanout": args.fanout,
        "chunk_len": args.chunk_len,
        "dim_toy_embed": args.dim,
        "max_length": args.max_length,
        "num_paths_total": num_paths_total,
        "num_paths_run": len(path_docs),
        "arms": arms,
        "git_sha": args.git_sha if args.git_sha else _git_short_sha(_REPO_ROOT),
        "torch_version": torch.__version__,
    }

    text = json.dumps(out, indent=2)
    print(text)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
