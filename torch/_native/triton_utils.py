import functools
import importlib
import importlib.metadata
import logging
import os

from .registry import register_op_registerer, _RegisterFn
from .common_utils import (
    check_native_jit_disabled,
    _available_version,
    _unavailable_reason,
)

log = logging.getLogger(__name__)

_TRITON_AVAILABLE = None
_TRITON_VERSION = None

@functools.cache
def _check_runtime_available() -> bool:
    """
    Check if triton is available

    NOTE: Doesn't import at this point
    """
    global _TRITON_AVAILABLE
    global _TRITON_VERSION

    if _TRITON_AVAILABLE is not None:
        return _TRITON_AVAILABLE

    deps = [("triton", "triton"), ]
    reason = _unavailable_reason(deps)
    if reason is None:
        _TRITON_AVAILABLE = True
        _TRITON_VERSION = _available_version("triton")
    else:
        print(
            "triton native DSL ops require: "
            "`triton`"
            f"{reason}"
        )
        _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE

_check_runtime_available()

def runtime_available() -> bool:
    return _TRITON_AVAILABLE

def runtime_version() -> None | tuple[int, int, int]:
    return _TRITON_VERSION

def register_op(fn: _RegisterFn) -> None:
    if (not _TRITON_AVAILABLE) or check_native_jit_disabled():
        return

    register_op_registerer(fn)
