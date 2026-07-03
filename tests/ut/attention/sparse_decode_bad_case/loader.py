# SPDX-License-Identifier: Apache-2.0
"""Load sparse-decode dump fixtures for layer-003 w8a8 bad-case UT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CASE_DIR = Path(__file__).resolve().parent
LAYER_003_DIR = "layer_003"
CALL_0000 = "call_0000.pt"


def sparse_decode_dump_path(variant: str) -> Path:
    """Return the path to the layer-003 call_0000 dump for ``variant``."""
    return CASE_DIR / variant / LAYER_003_DIR / CALL_0000


def load_sparse_decode_dump(
    variant: str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a sparse-decode dump payload saved by ``msa_m3_sparse_decode_dump``."""
    path = sparse_decode_dump_path(variant)
    if not path.is_file():
        raise FileNotFoundError(f"sparse decode dump not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def sparse_decode_inputs_from_dump(
    dump: dict[str, Any],
    *,
    device: str | torch.device,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    float,
    int,
]:
    """Unpack dump inputs and move tensors to ``device``."""
    inputs = dump["inputs"]
    q = inputs["q"].to(device)
    k_cache = inputs["k_cache"].to(device)
    v_cache = inputs["v_cache"].to(device)
    kv_cache = (k_cache, v_cache)
    topk_idx = inputs["topk_idx"].to(device)
    block_table = inputs["block_table"].to(device)
    seq_lens = inputs["seq_lens"].to(device)
    num_kv_heads = int(inputs["num_kv_heads"])
    sm_scale = float(inputs["sm_scale"])
    decode_query_len = int(inputs["decode_query_len"])
    return (
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        decode_query_len,
    )
