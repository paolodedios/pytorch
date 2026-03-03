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

_TRITON_AVAILABLE = None
_TRITON_VERSION = None

_BLESSED_VERSIONS: set[tuple[int, int, int]] = {
    (3, 6, 0),
}


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

    deps = [
        ("triton", "triton"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        _TRITON_AVAILABLE = True
        _TRITON_VERSION = _available_version("triton")
    else:
        print(f"triton native DSL ops require: `triton`{reason}")
        _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


_check_runtime_available()


def runtime_available() -> bool:
    if not _TRITON_AVAILABLE:
        return False

    return _TRITON_AVAILABLE


def runtime_version() -> None | tuple[int, int, int]:
    return _TRITON_VERSION


def _version_is_blessed() -> bool:
    if check_native_version_skip():
        return True
    if _TRITON_VERSION is None:
        return False
    return _TRITON_VERSION in _BLESSED_VERSIONS


def register_op(fn: _RegisterFn) -> None:
    if (not _TRITON_AVAILABLE) or check_native_jit_disabled():
        return

    if not _version_is_blessed():
        log.warning(
            "triton version %s is not blessed (blessed: %s); "
            "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
            _TRITON_VERSION,
            _BLESSED_VERSIONS,
        )
        return

    register_op_registerer(fn)
