#!/usr/bin/env python3
"""
S3 / §7.2 **TF-KV**：与 S1/S2 相同的玩具树、**单条根—叶路径**；使用**预层归一化**因果 Transformer
 trunk（``d_model`` / ``nhead`` / ``layers`` / ``ff_mult`` 与 path reader 默认一致），按 **chunk** 增量前向并
 **缓存每层 MHA 的 K/V**。

每边界 *k* 报告：

- ``kv_cache_nbytes``：当前已处理 ``(k+1)*chunk_len`` 个 token 后，**所有层** K+V 总字节数；
- ``increment_last_chunk_mean_ms``：在缓存已含前 *k* 个节点的前提下，**仅前向第 *k+1* 个节点 chunk**
  的平均 wall-clock（CUDA 同步；含 ``peak_alloc_mib`` 于该 micro-benchmark 内）。

可选 ``--branch-truncate-demo``：在根下走错子 **chunk**、``truncate_kv`` 回到仅根、再喂**兄弟**子首 chunk，
并记录截断耗时与各步 KV 字节（对应「丢弃子分支 KV 后缀 → 续算兄弟」的玩具复现）。

**与 ``TransformerPathReader``**：后者为 ``nn.TransformerEncoder``（**非**增量 KV）。本脚本 trunk 为手写因果
MHA+FFN（Pre-LN），参数量级与宽度一致，用于 **KV 协议**测量，非逐算子对齐 HF。

  python scripts/research/benchmark_tf_kv_path_segments.py --device cuda
  python scripts/research/benchmark_tf_kv_path_segments.py --device cuda --branch-truncate-demo
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

import torch
import torch.nn as nn

from src.rag_tree.tree import build_balanced_tree, iter_root_leaf_paths


def _git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak_mib(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mib(dev: torch.device) -> float:
    if dev.type != "cuda":
        return 0.0
    _sync(dev)
    return float(torch.cuda.max_memory_allocated()) / (1024**2)


class _CausalLayerCached(nn.Module):
    """Pre-LN causal self-attn + FFN; MHA K/V cached per token prefix for incremental chunks."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout_p = dropout

    def forward_chunk(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        pos_offset: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, Tc, D]; pos_offset = number of tokens before this chunk
        B, Tc, D = x.shape
        H, Dh = self.nhead, self.head_dim
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, Tc, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]
        if past_kv is None:
            K = k_new
            V = v_new
        else:
            Kp, Vp = past_kv
            K = torch.cat([Kp, k_new], dim=2)
            V = torch.cat([Vp, v_new], dim=2)
        scale = Dh**-0.5
        outs: list[torch.Tensor] = []
        for i in range(Tc):
            qi = q[:, :, i : i + 1, :]
            k_len = pos_offset + i + 1
            Ks = K[:, :, :k_len, :]
            Vs = V[:, :, :k_len, :]
            scores = (qi @ Ks.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            outs.append(attn @ Vs)
        o = torch.cat(outs, dim=2).transpose(1, 2).contiguous().reshape(B, Tc, D)
        x = x + self.out_proj(o)
        x = x + self.ff(self.norm2(x))
        return x, (K, V)


class IncrementalCausalTransformerKV(nn.Module):
    def __init__(self, dim: int, nhead: int, num_layers: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        dim_ff = dim * ff_mult
        self.layers = nn.ModuleList(
            [_CausalLayerCached(dim, nhead, dim_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self._past: list[tuple[torch.Tensor, torch.Tensor] | None]

    def reset(self) -> None:
        self._past = [None] * len(self.layers)

    def kv_nbytes(self) -> int:
        total = 0
        for past in self._past:
            if past is None:
                continue
            K, V = past
            total += int(K.numel() * K.element_size() + V.numel() * V.element_size())
        return total

    def truncate_kv(self, keep_tokens: int) -> None:
        for li, past in enumerate(self._past):
            if past is None:
                continue
            K, V = past
            if keep_tokens > K.shape[2]:
                raise ValueError(f"truncate keep_tokens={keep_tokens} > cache len {K.shape[2]}")
            self._past[li] = (
                K[:, :, :keep_tokens, :].contiguous(),
                V[:, :, :keep_tokens, :].contiguous(),
            )

    def forward_chunk(self, x: torch.Tensor, pos_offset: int) -> torch.Tensor:
        for li, layer in enumerate(self.layers):
            x, kv = layer.forward_chunk(x, self._past[li], pos_offset)
            self._past[li] = kv
        return x


def _fill_prefix(model: IncrementalCausalTransformerKV, chunks: list[torch.Tensor], upto: int) -> int:
    """Apply chunks[0:upto]; return token count."""
    model.reset()
    pos = 0
    for j in range(upto):
        c = chunks[j].unsqueeze(0)
        model.forward_chunk(c, pos_offset=pos)
        pos += int(c.shape[1])
    return pos


def _increment_last_chunk_stats(
    model: IncrementalCausalTransformerKV,
    chunks: list[torch.Tensor],
    seg_i: int,
    dev: torch.device,
    warmup: int,
    reps: int,
) -> tuple[float, float]:
    """Time only forward_chunk(chunks[seg_i]) with cache prefilled by chunks[0:seg_i]."""
    model.eval()
    model.to(dev)
    if seg_i == 0:
        upto = 0
    else:
        upto = seg_i
    chunk = chunks[seg_i].unsqueeze(0).to(dev)

    def run_once() -> None:
        _fill_prefix(model, chunks, upto)
        pos = upto * int(chunks[0].shape[0])
        with torch.no_grad():
            model.forward_chunk(chunk, pos_offset=pos)

    with torch.no_grad():
        for _ in range(warmup):
            run_once()
    _sync(dev)
    _reset_peak_mib(dev)
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            run_once()
    _sync(dev)
    mean_ms = (time.perf_counter() - t0) / max(reps, 1) * 1000.0
    return mean_ms, _peak_mib(dev)


def _branch_truncate_demo(
    root,
    *,
    chunk_len: int,
    model: IncrementalCausalTransformerKV,
    dev: torch.device,
    warmup: int,
    reps: int,
) -> dict[str, float | int]:
    """Wrong first child chunk → truncate to root → sibling first chunk."""
    if root.is_leaf() or len(root.children) < 2:
        return {"error": "need fanout>=2 and non-leaf root"}

    wrong = root.children[0].embedding
    sibling = root.children[1].embedding
    root_e = root.embedding
    assert int(root_e.shape[0]) == chunk_len

    model.eval()
    model.to(dev)

    def run_truncate() -> None:
        model.truncate_kv(chunk_len)

    with torch.no_grad():
        model.reset()
        model.forward_chunk(root_e.unsqueeze(0), pos_offset=0)
        b0 = model.kv_nbytes()

        model.forward_chunk(wrong.unsqueeze(0), pos_offset=chunk_len)
        b1 = model.kv_nbytes()

        for _ in range(warmup):
            model.reset()
            model.forward_chunk(root_e.unsqueeze(0), pos_offset=0)
            model.forward_chunk(wrong.unsqueeze(0), pos_offset=chunk_len)
            run_truncate()
            model.forward_chunk(sibling.unsqueeze(0), pos_offset=chunk_len)

        _sync(dev)
        t_trunc = 0.0
        for _ in range(reps):
            model.reset()
            model.forward_chunk(root_e.unsqueeze(0), pos_offset=0)
            model.forward_chunk(wrong.unsqueeze(0), pos_offset=chunk_len)
            _sync(dev)
            t0 = time.perf_counter()
            run_truncate()
            _sync(dev)
            t_trunc += time.perf_counter() - t0
        trunc_ms = t_trunc / max(reps, 1) * 1000.0

        model.reset()
        model.forward_chunk(root_e.unsqueeze(0), pos_offset=0)
        model.forward_chunk(wrong.unsqueeze(0), pos_offset=chunk_len)
        run_truncate()
        b2 = model.kv_nbytes()

        model.forward_chunk(sibling.unsqueeze(0), pos_offset=chunk_len)
        b3 = model.kv_nbytes()

    return {
        "kv_nbytes_after_root": b0,
        "kv_nbytes_after_wrong_child": b1,
        "kv_nbytes_after_truncate": b2,
        "kv_nbytes_after_sibling": b3,
        "truncate_mean_ms": round(trunc_ms, 6),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--branch-truncate-demo", action="store_true")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if args.dim % args.nhead != 0:
        print("dim must be divisible by nhead", file=sys.stderr)
        return 1

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)

    root = build_balanced_tree(args.depth, args.fanout, args.chunk_len, args.dim, dev, torch.float32, gen)
    paths = list(iter_root_leaf_paths(root))
    path = paths[0]
    nodes = len(path)
    chunks = [n.embedding for n in path]

    model = IncrementalCausalTransformerKV(
        dim=args.dim,
        nhead=args.nhead,
        num_layers=args.tf_layers,
        ff_mult=args.ff_mult,
    ).to(dev)

    per_seg: list[dict[str, float | int]] = []
    for seg_i in range(len(chunks)):
        _fill_prefix(model, chunks, seg_i + 1)
        kv_bytes = model.kv_nbytes()
        inc_ms, peak_mib = _increment_last_chunk_stats(
            model, chunks, seg_i, dev, args.warmup, args.reps
        )
        seq_len = (seg_i + 1) * args.chunk_len
        per_seg.append(
            {
                "segment_index": seg_i,
                "nodes_on_path": nodes,
                "seq_len": seq_len,
                "kv_cache_nbytes": kv_bytes,
                "increment_last_chunk_mean_ms": round(inc_ms, 6),
                "peak_alloc_mib": round(peak_mib, 4) if dev.type == "cuda" else 0.0,
            }
        )

    payload: dict[str, object] = {
        "kind": "tf_kv_path_segments",
        "baseline": "TF-KV",
        "git_sha": _git_short_sha(_REPO_ROOT),
        "device": str(dev),
        "tree_depth_param": args.depth,
        "path_nodes": nodes,
        "chunk_len": args.chunk_len,
        "dim": args.dim,
        "tf_layers": args.tf_layers,
        "nhead": args.nhead,
        "ff_mult": args.ff_mult,
        "warmup": args.warmup,
        "reps": args.reps,
        "per_segment": per_seg,
        "note": (
            "Pre-LN causal trunk with per-layer MHA KV cache; per row: state after nodes 0..k; "
            "increment_last_chunk_mean_ms = time to run only chunk k given cache for 0..k-1. "
            "§7.2 TF-KV KV bytes + incremental continue cost (linear path). "
            "Optional --branch-truncate-demo for wrong-child → truncate → sibling."
        ),
    }

    if args.branch_truncate_demo:
        payload["branch_truncate_demo"] = _branch_truncate_demo(
            root,
            chunk_len=args.chunk_len,
            model=model,
            dev=dev,
            warmup=args.warmup,
            reps=args.reps,
        )

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
