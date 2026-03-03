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

_CUTEDSL_AVAILABLE = None
_CUTEDSL_VERSION = None

log = logging.getLogger(__name__)

@functools.cache
def _check_runtime_available() -> bool:
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    global _CUTEDSL_AVAILABLE
    global _CUTEDSL_VERSION

    if _CUTEDSL_AVAILABLE is not None:
        return _CUTEDSL_AVAILABLE

    deps = [("nvidia_cutlass_dsl", "cutlass"),
            ("apache_tvm_ffi", "tvm_ffi"),
            ("cuda_bindings", "cuda.bindings.driver"),
           ]
    reason = _unavailable_reason(deps)
    if reason is None:
        _CUTEDSL_AVAILABLE = True
        _CUTEDSL_VERSION = _available_version("nvidia_cutlass_dsl")
    else:
        print(
            "CuTeDSL operators require optional Python packages "
            "`nvidia-cutlass-dsl`, `apache-tvm-ffi`, and `cuda-bindings` "
            "(from NVIDIA cuda-python); "
            f"{reason}"
        )
        _CUTEDSL_AVAILABLE = False
    return _CUTEDSL_AVAILABLE

_check_runtime_available()

def runtime_available() -> bool:
    return _CUTEDSL_AVAILABLE

def runtime_version() -> None | tuple[int, int, int]:
    return _CUTEDSL_VERSION

def register_op(fn: _RegisterFn) -> None:
    if (not _CUTEDSL_AVAILABLE) or check_native_jit_disabled():
        log.info(f'{__name__} not registering native ops')
        return

    register_op_registerer(fn)
