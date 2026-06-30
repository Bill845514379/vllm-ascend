# SPDX-License-Identifier: Apache-2.0
"""Runtime Triton vs torch precision checks for MiniMax M3 sparse attention."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)

# Match tests/ut/attention/test_minimax_m3_sparse_attn.py thresholds.
_SPARSE_MEAN_ATOL = 2.5e-4
_SPARSE_MAX_ATOL = 1.7e-2

_DEBUG_ENABLED = os.environ.get("VLLM_ASCEND_MINIMAX_M3_SPARSE_DEBUG", "0") == "1"
_DEBUG_MAX = int(os.environ.get("VLLM_ASCEND_MINIMAX_M3_SPARSE_DEBUG_MAX", "50"))

_check_count = 0
_fail_count = 0
_logged_config = False


def sparse_debug_enabled() -> bool:
    return _DEBUG_ENABLED


def _log_debug_config_once() -> None:
    global _logged_config
    if _logged_config or not _DEBUG_ENABLED:
        return
    logger.warning(
        "MiniMax M3 sparse attn precision debug enabled "
        "(max_checks=%d, mean_atol=%.6g, max_atol=%.6g)",
        _DEBUG_MAX,
        _SPARSE_MEAN_ATOL,
        _SPARSE_MAX_ATOL,
    )
    _logged_config = True


def _should_check() -> bool:
    global _check_count
    if not _DEBUG_ENABLED:
        return False
    if _check_count >= _DEBUG_MAX:
        return False
    if get_forward_context().capturing:
        return False
    return True


def _synchronize() -> None:
    if torch.npu.is_available():
        torch.npu.synchronize()


def _format_tensor_summary(name: str, tensor: torch.Tensor, max_elems: int = 8) -> str:
    flat = tensor.detach().reshape(-1)
    n = min(max_elems, flat.numel())
    sample = flat[:n].float().cpu().tolist()
    return (
        f"{name}=shape{tuple(tensor.shape)} dtype={tensor.dtype} "
        f"sample={sample}"
    )


def _compare_and_log(
    op_name: str,
    triton_out: torch.Tensor,
    torch_out: torch.Tensor,
    case: dict[str, Any],
) -> None:
    global _check_count, _fail_count
    _check_count += 1

    error = (triton_out.float() - torch_out.float()).abs()
    mean_err = error.mean().item()
    max_err = error.max().item()
    passed = mean_err < _SPARSE_MEAN_ATOL and max_err < _SPARSE_MAX_ATOL

    if passed:
        logger.info(
            "MiniMax M3 sparse attn precision OK [%s] check=%d/%d "
            "mean_err=%.6g max_err=%.6g layer=%s",
            op_name,
            _check_count,
            _DEBUG_MAX,
            mean_err,
            max_err,
            case.get("layer_name", "?"),
        )
        return

    _fail_count += 1
    flat_err = error.reshape(-1)
    worst_idx = int(flat_err.argmax().item())
    triton_flat = triton_out.reshape(-1)
    torch_flat = torch_out.reshape(-1)

    logger.warning(
        "MiniMax M3 sparse attn precision MISMATCH [%s] "
        "check=%d/%d fail=%d mean_err=%.6g (tol=%.6g) "
        "max_err=%.6g (tol=%.6g) layer=%s "
        "num_decodes=%s num_prefills=%s decode_query_len=%s "
        "q_shape=%s topk_shape=%s seq_lens=%s block_table_shape=%s "
        "worst_idx=%d triton=%.6g torch=%.6g diff=%.6g "
        "triton_sample=%s torch_sample=%s topk_sample=%s",
        op_name,
        _check_count,
        _DEBUG_MAX,
        _fail_count,
        mean_err,
        _SPARSE_MEAN_ATOL,
        max_err,
        _SPARSE_MAX_ATOL,
        case.get("layer_name", "?"),
        case.get("num_decodes"),
        case.get("num_prefills"),
        case.get("decode_query_len"),
        case.get("q_shape"),
        case.get("topk_shape"),
        case.get("seq_lens"),
        case.get("block_table_shape"),
        worst_idx,
        triton_flat[worst_idx].float().item(),
        torch_flat[worst_idx].float().item(),
        flat_err[worst_idx].item(),
        case.get("triton_sample"),
        case.get("torch_sample"),
        case.get("topk_sample"),
    )


def verify_sparse_attn_decode(
    *,
    layer_name: str,
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    triton_out: torch.Tensor,
    decode_query_len: int,
    num_decodes: int,
    num_prefills: int,
) -> None:
    _log_debug_config_once()
    if not _should_check():
        return

    from vllm_ascend.attention.msa_m3_ops import minimax_m3_sparse_attn_decode_torch

    torch_out = torch.empty_like(triton_out)
    minimax_m3_sparse_attn_decode_torch(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        torch_out,
        decode_query_len,
    )
    _synchronize()
    _compare_and_log(
        "minimax_m3_sparse_attn_decode",
        triton_out,
        torch_out,
        {
            "layer_name": layer_name,
            "num_decodes": num_decodes,
            "num_prefills": num_prefills,
            "decode_query_len": decode_query_len,
            "q_shape": tuple(q.shape),
            "topk_shape": tuple(topk_idx.shape),
            "seq_lens": seq_lens.detach().cpu().tolist(),
            "block_table_shape": tuple(block_table.shape),
            "triton_sample": triton_out.detach().float().reshape(-1)[:8].cpu().tolist(),
            "torch_sample": torch_out.detach().float().reshape(-1)[:8].cpu().tolist(),
            "topk_sample": topk_idx.detach().cpu().reshape(-1)[:16].tolist(),
        },
    )


def verify_sparse_attn_prefill(
    *,
    layer_name: str,
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    triton_out: torch.Tensor,
    num_decodes: int,
    num_prefills: int,
) -> None:
    _log_debug_config_once()
    if not _should_check():
        return

    from vllm_ascend.attention.msa_m3_ops import minimax_m3_sparse_attn_torch

    torch_out = torch.empty_like(triton_out)
    minimax_m3_sparse_attn_torch(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        torch_out,
    )
    _synchronize()
    _compare_and_log(
        "minimax_m3_sparse_attn",
        triton_out,
        torch_out,
        {
            "layer_name": layer_name,
            "num_decodes": num_decodes,
            "num_prefills": num_prefills,
            "max_query_len": max_query_len,
            "q_shape": tuple(q.shape),
            "topk_shape": tuple(topk_idx.shape),
            "seq_lens": seq_lens.detach().cpu().tolist(),
            "prefix_lens": prefix_lens.detach().cpu().tolist(),
            "cu_seqlens_q": cu_seqlens_q.detach().cpu().tolist(),
            "block_table_shape": tuple(block_table.shape),
            "triton_sample": triton_out.detach().float().reshape(-1)[:8].cpu().tolist(),
            "torch_sample": torch_out.detach().float().reshape(-1)[:8].cpu().tolist(),
            "topk_sample": topk_idx.detach().cpu().reshape(-1)[:16].tolist(),
        },
    )
