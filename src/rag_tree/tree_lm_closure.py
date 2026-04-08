"""根—叶路径文本 → HuggingFace **因果 LM**（含 **lm_head**）：teacher-forcing 损失与续写生成。

与 ``Mamba2PathReader`` 等 **path-batch 嵌入基准**分离：此处使用 **真 tokenizer + 词表**，节点 ``TreeNode.text`` 拼接成文档。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from src.rag_tree.tree import TreeNode, iter_root_leaf_paths


def path_to_document(path: Sequence[TreeNode], *, sep: str = "\n\n") -> str:
    """将路径上各节点 ``text`` 非空行用分隔符拼成一条 UTF-8 文档。"""
    parts = [(n.text or "").strip() for n in path]
    parts = [p for p in parts if p]
    return sep.join(parts)


def iter_path_documents(root: TreeNode, *, sep: str = "\n\n") -> List[Tuple[List[TreeNode], str]]:
    """所有根—叶路径及其拼接文档（顺序与 ``iter_root_leaf_paths`` 一致）。"""
    out: List[Tuple[List[TreeNode], str]] = []
    for path in iter_root_leaf_paths(root):
        out.append((path, path_to_document(path, sep=sep)))
    return out


def ensure_causal_lm_tokenizer(tokenizer) -> None:
    """GPT-2 系常无 ``pad_token``；生成与 batch 对齐时需落到 ``eos``。"""
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


@torch.inference_mode()
def causal_lm_loss_for_document(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_length: int = 512,
) -> torch.Tensor:
    """单文档 next-token CE（HF ``labels=input_ids`` 语义）。空串返回 NaN 标量。"""
    if not text.strip():
        return torch.tensor(float("nan"), device=device, dtype=torch.float32)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, labels=enc["input_ids"])
    if out.loss is None:
        raise RuntimeError("model returned no loss; need AutoModelForCausalLM with labels=")
    return out.loss.detach()


def causal_lm_mean_loss_on_documents(
    model: torch.nn.Module,
    tokenizer,
    documents: Sequence[str],
    device: torch.device,
    *,
    max_length: int = 512,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """对多条路径文档逐条算 loss，返回 (均值, 逐条列表)；跳过空文档。"""
    losses: List[torch.Tensor] = []
    for doc in documents:
        if not (doc or "").strip():
            continue
        losses.append(causal_lm_loss_for_document(model, tokenizer, doc, device, max_length=max_length))
    if not losses:
        return torch.tensor(float("nan"), device=device), []
    stacked = torch.stack(losses)
    return stacked.mean(), losses


@torch.inference_mode()
def generate_continuation(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_new_tokens: int = 16,
    max_context_length: int = 512,
) -> str:
    """在整段路径文本后贪心续写 ``max_new_tokens`` 个 token（``do_sample=False``）。"""
    ensure_causal_lm_tokenizer(tokenizer)
    if not text.strip():
        return ""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_length,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def train_one_step_mean_loss(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    documents: Sequence[str],
    device: torch.device,
    *,
    max_length: int = 512,
    max_grad_norm: float = 1.0,
) -> float:
    """对非空文档逐条前向，对 **batch 平均 CE** 一次 ``backward`` + ``optimizer.step()``（最小训练闭环）。"""
    docs = [(d or "").strip() for d in documents if (d or "").strip()]
    if not docs:
        return float("nan")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[torch.Tensor] = []
    for text in docs:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        if out.loss is None:
            raise RuntimeError("model returned no loss")
        losses.append(out.loss)
    loss_mean = torch.stack(losses).mean()
    loss_mean.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss_mean.detach().cpu().item())
