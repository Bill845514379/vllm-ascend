#!/usr/bin/env python3
"""Shrink an online sparse-decode golden dump for checking into the repo.

Usage (after online serve with debug dump enabled)::

    /usr/local/python3.11.10/bin/python tests/ut/attention/fixtures/build_bundled_golden.py \\
        /path/to/minimax_m3_decode_L3_S138_DQ1_F1.pt

Writes ``bundled/minimax_m3_decode_L3_S138_DQ1_F1.pt`` (~200KB). Only the
bundled file is checked in; full dumps can be deleted after bundling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_bundled(src: Path, dst: Path) -> None:
    payload = torch.load(src, map_location="cpu")
    block_table = payload["block_table"]
    used_pages = sorted(
        {int(x.item()) for x in block_table.reshape(-1) if int(x.item()) >= 0}
    )
    pool_blocks = max(used_pages) + 1 if used_pages else 1
    bundled = {
        "case_id": payload.get("case_id", dst.stem),
        "layer_name": payload.get("layer_name"),
        "q": payload["q"].clone(),
        "key_cache": payload["key_cache"][:pool_blocks].clone(),
        "value_cache": payload["value_cache"][:pool_blocks].clone(),
        "topk_idx": payload["topk_idx"].clone(),
        "block_table": payload["block_table"].clone(),
        "seq_lens": payload["seq_lens"].clone(),
        "triton_out": payload["triton_out"].clone(),
        "torch_out": payload["torch_out"].clone(),
        "decode_query_len": int(payload["decode_query_len"]),
        "max_seq_len": int(payload["max_seq_len"]),
        "num_kv_heads": int(payload["num_kv_heads"]),
        "num_q_heads": int(payload["num_q_heads"]),
        "sm_scale": float(payload["sm_scale"]),
        "bundled_from": src.name,
        "kv_pool_blocks": pool_blocks,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundled, dst, _use_new_zipfile_serialization=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        type=Path,
        help="full online debug dump .pt to shrink",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parent / "bundled" / "minimax_m3_decode_L3_S138_DQ1_F1.pt",
    )
    args = parser.parse_args()
    build_bundled(args.src, args.dst)
    print(f"wrote {args.dst} ({args.dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
