# SPDX-License-Identifier: Apache-2.0
"""Debug logging for AFD attention batch / padding / CAM wire (runner + pad).

Set ``VLLM_ASCEND_AFD_ATTN_WIRE_LOG=1``. Safe for short runs; uses ``print`` only.
"""
from __future__ import annotations

import os
from typing import Any, Optional

_TAG = "[AFD-ATTN-WIRE]"


def is_afd_attn_wire_log_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_AFD_ATTN_WIRE_LOG", "0") == "1"


def _nta_list(nta: Any) -> Any:
    if nta is None:
        return None
    try:
        return nta.detach().cpu().tolist()
    except Exception:
        try:
            return nta.tolist()
        except Exception:
            return repr(nta)


def _bd_short(bd: Any) -> str:
    if bd is None:
        return "None"
    try:
        nt = getattr(bd, "num_tokens", None)
        nr = getattr(bd, "num_reqs", None)
        un = getattr(bd, "uniform", None)
        return f"BD(num_tokens={nt},num_reqs={nr},uniform={un})"
    except Exception:
        return repr(bd)[:120]


def format_nta(x: Any) -> Any:
    """Public alias for logging ``num_tokens_across_dp`` tensors."""
    return _nta_list(x)


def format_batch_descriptor(bd: Any) -> str:
    return _bd_short(bd)


def hidden_states_row0(hs: Any) -> Any:
    """Best-effort batch row count from model output (Tensor, tuple, IntermediateTensors)."""
    if hs is None:
        return None
    shape = getattr(hs, "shape", None)
    if shape is not None and len(shape) > 0:
        try:
            return int(shape[0])
        except Exception:
            pass
    if isinstance(hs, tuple) and len(hs) > 0:
        return hidden_states_row0(hs[0])
    if isinstance(hs, list) and len(hs) > 0:
        return hidden_states_row0(hs[0])
    tens = getattr(hs, "tensors", None)
    if isinstance(tens, dict):
        for key in ("hidden_states", "last_hidden"):
            if key in tens:
                return hidden_states_row0(tens[key])
        if tens:
            return hidden_states_row0(next(iter(tens.values())))
    return f"?{type(hs).__name__}"


def _safe_extra_val(v: Any) -> str:
    """Avoid ``str(tensor)`` on device tensors (can error or sync oddly)."""
    try:
        sh = getattr(v, "shape", None)
        if sh is not None and len(sh) > 0:
            dt = getattr(v, "dtype", None)
            return f"Tensor(shape={tuple(sh)},dtype={dt})"
    except Exception:
        pass
    try:
        s = repr(v)
        return s if len(s) <= 240 else s[:237] + "..."
    except Exception:
        return "<?>"


def log_afd_attn_wire(
    tag: str,
    ctx: Any | None = None,
    **extra: Any,
) -> None:
    """One-line snapshot: optional ``ForwardContext`` + arbitrary extra key=values."""
    if not is_afd_attn_wire_log_enabled():
        return
    try:
        parts: list[str] = [_TAG, tag]
        dp_r: Any = None
        try:
            from vllm.distributed.parallel_state import get_dp_group

            g = get_dp_group()
            if g is not None:
                dp_r = int(g.rank_in_group)
        except Exception:
            pass
        parts.append(f"dp_rank={dp_r}")
        if ctx is not None:
            parts.append(f"afd_expected={getattr(ctx, 'afd_expected_a2e_rows', None)}")
            parts.append(f"ctx_num_tokens={getattr(ctx, 'num_tokens', None)}")
            parts.append(f"ubatch_idx={getattr(ctx, 'ubatch_idx', None)}")
            parts.append(f"num_ubatches={getattr(ctx, 'num_ubatches', None)}")
            us = getattr(ctx, "ubatch_slices", None)
            parts.append(f"ubatch_slices={'yes' if us is not None else 'no'}")
            parts.append(f"cgrm={getattr(ctx, 'cudagraph_runtime_mode', None)}")
            dm = getattr(ctx, "dp_metadata", None)
            if dm is not None:
                parts.append(f"dp_nta={_nta_list(getattr(dm, 'num_tokens_across_dp_cpu', None))}")
            parts.append(f"bd={_bd_short(getattr(ctx, 'batch_descriptor', None))}")
        for k in sorted(extra.keys()):
            parts.append(f"{k}={_safe_extra_val(extra[k])}")
        print(" ".join(str(p) for p in parts), flush=True)
    except Exception as e:
        try:
            print(f"{_TAG} {tag} log_failed={e!r}", flush=True)
        except Exception:
            pass


def log_afd_attn_wire_pad(
    *,
    tag: str,
    row0: int,
    expected: Optional[int],
    expected_src: str,
    compute_gate: int,
    action: str,
    pad_rows: int = 0,
    new_row0: Optional[int] = None,
) -> None:
    if not is_afd_attn_wire_log_enabled():
        return
    try:
        msg = (
            f"{_TAG} {tag} row0={row0} expected={expected} "
            f"expected_src={expected_src} compute_gate={compute_gate} "
            f"action={action} pad_rows={pad_rows} new_row0={new_row0}"
        )
        print(msg, flush=True)
    except Exception:
        pass
