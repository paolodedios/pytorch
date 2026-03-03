import functools
import logging

from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .registry import _RegisterFn, register_op_registerer


log = logging.getLogger(__name__)

_CUTEDSL_AVAILABLE = None
_CUTEDSL_VERSION = None

_BLESSED_VERSIONS: set[tuple[int, int, int]] = {
    (4, 4, 1),
}


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

    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
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
    if not _CUTEDSL_AVAILABLE:
        return False

    return _CUTEDSL_AVAILABLE


def runtime_version() -> None | tuple[int, int, int]:
    return _CUTEDSL_VERSION


def _version_is_blessed() -> bool:
    if check_native_version_skip():
        return True
    if _CUTEDSL_VERSION is None:
        return False
    return _CUTEDSL_VERSION in _BLESSED_VERSIONS


def register_op(fn: _RegisterFn) -> None:
    if (not _CUTEDSL_AVAILABLE) or check_native_jit_disabled():
        log.info("%s not registering native ops", __name__)
        return

    if not _version_is_blessed():
        log.warning(
            "cutedsl version %s is not blessed (blessed: %s); "
            "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
            _CUTEDSL_VERSION,
            _BLESSED_VERSIONS,
        )
        return

    register_op_registerer(fn)
