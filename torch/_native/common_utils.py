import importlib
import importlib.metadata
import os

from functools import cache

@cache
def check_native_jit_disabled() -> bool:
    """
    Single point to check if native DSL ops are disabled globally,
    checked via:
    TORCH_DISABLE_NATIVE_JIT=1
    """
    return int(os.getenv("TORCH_DISABLE_NATIVE_JIT", 0)) == 1


def _unavailable_reason(deps: list[tuple[str, str]]) -> None | str:
    """
    Check availability of required packages - cuteDSL & deps,
    informing user what (if anything) is missing

    NOTE: Doesn't actually import anything.
    """
    for package_name, module_name in deps:
        # This doesn't actually import the packages, to reduce
        # overall import time & memory.
        if importlib.util.find_spec(module_name) is None:
            return (
                f"missing optional dependency `{package_name}` "
                f"(importlib.util.find_spec({package_name}) failed)"
            )
    return None

def _available_version(package: str) -> tuple[int, int, int]:
    """
    Get version of the installed "nvidia-cutlass-dsl" package

    Assumes the package exists, i.e. will fail if it doesn't
    """
    version = importlib.metadata.version(package)

    major, minor, update = (int(vi) for vi in version.split("."))

    return (major, minor, update)


