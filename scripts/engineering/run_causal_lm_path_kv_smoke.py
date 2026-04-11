#!/usr/bin/env python3
"""
Sprint 2 工程烟测：**同 `benchmark_wikitext_tree` 建树** → 根—叶路径 **文档** → **HF 因果 LM** 单次前向（**use_cache=True**），记录
**`torch.cuda.max_memory_allocated` 峰值** 与 **cache / past 张量 nbytes**。

默认臂：**`openai-community/gpt2`**。可选 Mamba2（**`--mamba`**）：默认 **`AntonV/mamba2-370m-hf`**（含 ``model_type``、tokenizer、``safetensors``，可被 ``AutoModelForCausalLM`` 加载）。显存不足时用 **`--gpt2-only`**。

与 **path-batch 玩具 trunk**（`Mamba2PathReader`）**分列** —— 见 **`ENGINEERING_NORTH_STAR_PLAN.md` §4.1**。

  py -3 scripts/engineering/run_causal_lm_path_kv_smoke.py --out-json results/metrics_result/engineering/eng_causal_kv_gpt2.json
  py -3 scripts/engineering/run_causal_lm_path_kv_smoke.py --mamba --out-json results/metrics_result/engineering/eng_causal_kv_both.json

**旧检查点** ``state-spaces/mamba2-370m``：无 ``model_type``、无标准 ``tokenizer.*``，与当前 ``transformers`` 的 ``Mamba2Config`` 不对齐；若仍要使用该 id，请自行换用兼容镜像或 **``--mamba-id``** 指向带 ``model_type: "mamba2"`` 的仓库。脚本在 **仅** ``state-spaces/mamba2-370m`` 时会用 **``--gpt2-id``** 作为分词器（可用 **``--mamba-tokenizer-id``** 覆盖）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    """Prefer fast tokenizer; on Hub checkpoints that need SPM/tiktoken only for fast conversion, fall back to slow."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)


def _tokenizer_hub_id_for_arm(
    arm_key: str, model_id: str, gpt2_id: str, mamba_tokenizer_id: str | None
) -> str:
    """``state-spaces/mamba2-370m`` has no tokenizer files on Hub — use GPT-2 BPE unless overridden."""
    if arm_key != "mamba2_370m":
        return model_id
    if mamba_tokenizer_id is not None:
        return mamba_tokenizer_id
    if model_id == "state-spaces/mamba2-370m":
        return gpt2_id
    return model_id


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    from src.rag_tree.causal_lm_kv_stats import (
        cache_nbytes_from_outputs,
        estimate_attention_kv_nbytes_mha_stack,
    )
    from src.rag_tree.from_text import build_bottom_up_text_tree
    from src.rag_tree.hf_corpus import wikitext2_leaf_chunks
    from src.rag_tree.tree_lm_closure import ensure_causal_lm_tokenizer, iter_path_documents

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-leaves", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chars-per-leaf", type=int, default=600)
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128, help="tree embedding dim (toy); LM uses tokenizer text only")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-paths", type=int, default=0, help="0 = all root-leaf paths")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpt2-id", type=str, default="openai-community/gpt2")
    p.add_argument(
        "--mamba-id",
        type=str,
        default="AntonV/mamba2-370m-hf",
        help="HF Mamba2 checkpoint (default: AntonV mirror with model_type + tokenizer; not state-spaces/mamba2-370m)",
    )
    p.add_argument(
        "--mamba-tokenizer-id",
        type=str,
        default=None,
        help="Tokenizer Hub id when --mamba-id has no tokenizer (e.g. state-spaces/mamba2-370m → default --gpt2-id)",
    )
    p.add_argument("--mamba", action="store_true", help="also run Mamba2-370m arm (large download + VRAM)")
    p.add_argument("--gpt2-only", action="store_true", help="only GPT-2 (default if --mamba not set)")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--git-sha", type=str, default=None)
    args = p.parse_args()

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
    path_docs = iter_path_documents(root, sep="\n\n")
    num_paths_total = len(path_docs)
    if args.max_paths > 0:
        path_docs = path_docs[: args.max_paths]

    arms: dict = {}
    model_ids: list[tuple[str, str]] = [(args.gpt2_id, "gpt2")]
    if args.mamba and not args.gpt2_only:
        model_ids.append((args.mamba_id, "mamba2_370m"))
    if args.gpt2_only:
        model_ids = [(args.gpt2_id, "gpt2")]

    for model_id, key in model_ids:
        try:
            tok_id = _tokenizer_hub_id_for_arm(
                key, model_id, args.gpt2_id, args.mamba_tokenizer_id
            )
            tokenizer = _load_tokenizer(tok_id)
            ensure_causal_lm_tokenizer(tokenizer)
            dtype = torch.float32 if device.type == "cpu" else torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            model.eval()
            model.to(device)
        except Exception as e:
            arms[key] = {"model_id": model_id, "error": str(e)}
            continue

        path_rows = []
        for i, (_path, doc) in enumerate(path_docs):
            row: dict = {"path_index": i, "doc_chars": len(doc)}
            if not doc.strip():
                row["skip"] = "empty_document"
                path_rows.append(row)
                continue
            enc = tokenizer(
                doc,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            seq_len = int(enc["input_ids"].shape[1])
            row["seq_len"] = seq_len

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            with torch.inference_mode():
                outputs = model(**enc, use_cache=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                row["peak_alloc_mib"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            else:
                row["peak_alloc_mib"] = None

            nbytes = int(cache_nbytes_from_outputs(outputs))
            row["past_or_cache_nbytes"] = nbytes
            if nbytes == 0:
                es = int(next(model.parameters()).element_size())
                est = estimate_attention_kv_nbytes_mha_stack(
                    model.config, seq_len, batch=1, element_size=es
                )
                if est is not None:
                    row["past_or_cache_nbytes_estimated"] = est
            path_rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        arm_out: dict = {
            "model_id": model_id,
            "device": str(device),
            "dtype": str(dtype),
            "paths": path_rows,
        }
        tok_id = _tokenizer_hub_id_for_arm(
            key, model_id, args.gpt2_id, args.mamba_tokenizer_id
        )
        if tok_id != model_id:
            arm_out["tokenizer_id"] = tok_id
        arms[key] = arm_out

    out: dict = {
        "kind": "engineering_causal_lm_path_kv_smoke",
        "schema_version": 1,
        "runner": "scripts/engineering/run_causal_lm_path_kv_smoke.py",
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
