# SPDX-License-Identifier: Apache-2.0
"""Debug dump helpers for MiniMax-M3 sparse decode attention."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

from vllm_ascend.attention.msa_m3_triton import _split_triton_main_kv_cache

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_DUMP_COUNTERS: dict[str, int] = {}
_DUMP_LOGGED = False
_DUMP_PREFIX: str | None = None


def sparse_decode_use_torch() -> bool:
    return os.environ.get(
        "VLLM_ASCEND_MINIMAX_M3_SPARSE_DECODE_USE_TORCH", "0"
    ) in ("1", "true", "True")


def _sparse_decode_dump_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_MINIMAX_M3_SPARSE_DEBUG", "0") in (
        "1",
        "true",
        "True",
    )


def _sparse_decode_dump_dir() -> Path:
    variant = "torch" if sparse_decode_use_torch() else "triton"
    default_dir = f"/workspace/bench_log/sparse_decode_dump/{variant}"
    return Path(
        os.environ.get("VLLM_ASCEND_MINIMAX_M3_SPARSE_DEBUG_DIR", default_dir)
    )


def _sparse_decode_dump_max_per_layer() -> int:
    return max(0, int(os.environ.get("VLLM_ASCEND_MINIMAX_M3_SPARSE_DEBUG_MAX", "1")))


def _safe_path_name(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_") or "unknown"


def _sparse_decode_dump_prefix() -> str:
    global _DUMP_PREFIX
    if _DUMP_PREFIX is None:
        explicit = os.environ.get(
            "VLLM_ASCEND_MINIMAX_M3_SPARSE_DEBUG_PREFIX", ""
        ).strip()
        if explicit:
            _DUMP_PREFIX = _safe_path_name(explicit)
        else:
            _DUMP_PREFIX = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return _DUMP_PREFIX


def _dump_counter_key(prefix: str, layer_name: str) -> str:
    return f"{prefix}:{layer_name}"


def _extract_layer_idx(layer_name: str) -> int:
    match = _LAYER_RE.search(layer_name)
    return int(match.group(1)) if match else -1


def _is_device_zero(tensor: torch.Tensor) -> bool:
    device = tensor.device
    if device.type in ("npu", "cuda"):
        return device.index in (None, 0)
    return device.index in (None, 0)


def _to_cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu_clone(v) for v in value)
    return value


def maybe_dump_sparse_decode(
    *,
    layer_name: str,
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
) -> None:
    """Save sparse-decode inputs/output for one layer when debug env is set."""
    global _DUMP_LOGGED

    if not _sparse_decode_dump_enabled() or not _is_device_zero(q):
        return

    max_dumps = _sparse_decode_dump_max_per_layer()

    dump_prefix = _sparse_decode_dump_prefix()
    counter_key = _dump_counter_key(dump_prefix, layer_name)
    call_idx = _DUMP_COUNTERS.get(counter_key, 0)
    if call_idx >= max_dumps:
        return

    if not _DUMP_LOGGED:
        dump_dir = _sparse_decode_dump_dir()
        variant = "torch" if sparse_decode_use_torch() else "triton"
        logger.warning(
            "MiniMax M3 sparse decode dump enabled: variant=%s, prefix=%s, "
            "dir=%s, max_per_layer=%d, device=%s",
            variant,
            dump_prefix,
            dump_dir,
            max_dumps,
            q.device,
        )
        _DUMP_LOGGED = True

    k_cache, v_cache = _split_triton_main_kv_cache(kv_cache)
    layer_idx = _extract_layer_idx(layer_name)
    safe_layer_name = _safe_path_name(layer_name)
    dump_dir = _sparse_decode_dump_dir()
    layer_dir = (
        dump_dir
        / dump_prefix
        / f"layer_{layer_idx:03d}_{safe_layer_name}"
    )
    layer_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dump_prefix": dump_prefix,
        "layer_name": layer_name,
        "layer_idx": layer_idx,
        "call_idx": call_idx,
        "variant": "torch" if sparse_decode_use_torch() else "triton",
        "device": str(q.device),
        "inputs": {
            "q": _to_cpu_clone(q),
            "k_cache": _to_cpu_clone(k_cache),
            "v_cache": _to_cpu_clone(v_cache),
            "topk_idx": _to_cpu_clone(topk_idx),
            "block_table": _to_cpu_clone(block_table),
            "seq_lens": _to_cpu_clone(seq_lens),
            "num_kv_heads": num_kv_heads,
            "sm_scale": sm_scale,
            "decode_query_len": decode_query_len,
        },
        "output": _to_cpu_clone(output),
    }
    dump_path = layer_dir / f"call_{call_idx:04d}.pt"
    torch.save(payload, dump_path)
    _DUMP_COUNTERS[counter_key] = call_idx + 1
