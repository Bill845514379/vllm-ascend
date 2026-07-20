# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 model package."""

from vllm_ascend.models.minimax_m3 import modeling as _modeling

__all__ = []
for _name in dir(_modeling):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_modeling, _name)
        __all__.append(_name)

del _modeling, _name
