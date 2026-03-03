# Owner(s): ["module: unknown"]

import os
import subprocess
import sys
import textwrap

from torch.testing._internal.common_utils import run_tests, TestCase


def _subprocess_lastline(script, env=None):
    """Run script in a fresh interpreter and return the last line of stdout."""
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        env=env,
        stderr=subprocess.DEVNULL,
    ).decode("ascii").strip()
    return result.rsplit("\n", 1)[-1]


class TestNativeDSLOps(TestCase):
    """Tests for the torch._native DSL ops framework."""

    def test_consistent_helper_interface(self):
        """triton_utils and cutedsl_utils expose the same public API."""
        from torch._native import triton_utils, cutedsl_utils

        REQUIRED_METHODS = {"runtime_available", "runtime_version", "register_op"}

        for mod in (triton_utils, cutedsl_utils):
            public = {
                name
                for name in dir(mod)
                if not name.startswith("_")
            }
            self.assertTrue(
                REQUIRED_METHODS <= public,
                f"{mod.__name__} missing: {REQUIRED_METHODS - public}",
            )
            for name in REQUIRED_METHODS:
                self.assertTrue(callable(getattr(mod, name)))

        triton_public = {n for n in dir(triton_utils) if not n.startswith("_")}
        cute_public = {n for n in dir(cutedsl_utils) if not n.startswith("_")}
        self.assertEqual(triton_public, cute_public)

        for mod in (triton_utils, cutedsl_utils):
            self.assertIsInstance(mod.runtime_available(), bool)
            ver = mod.runtime_version()
            if ver is not None:
                self.assertIsInstance(ver, tuple)
                self.assertEqual(len(ver), 3)
                for v in ver:
                    self.assertIsInstance(v, int)

    def test_no_dsl_imports_after_import_torch(self):
        """import torch must not transitively import DSL runtimes.

        Note: cuda.bindings may appear because importlib.util.find_spec on
        nested modules (e.g. cuda.bindings.driver) imports parent packages
        as a side-effect.  We check only the primary DSL runtimes here.
        """
        script = textwrap.dedent("""\
            import sys
            import torch
            dsl_modules = ["triton", "cutlass", "tvm_ffi"]
            leaked = [m for m in dsl_modules if m in sys.modules]
            print(repr(leaked))
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "[]", f"DSL modules leaked on import torch: {result}")

    def test_check_native_jit_disabled_default(self):
        """TORCH_DISABLE_NATIVE_JIT unset -> check returns False."""
        script = textwrap.dedent("""\
            import os
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)
            from torch._native.common_utils import check_native_jit_disabled
            print(check_native_jit_disabled())
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "False")

    def test_check_native_jit_disabled_set(self):
        """TORCH_DISABLE_NATIVE_JIT=1 -> check returns True."""
        script = textwrap.dedent("""\
            from torch._native.common_utils import check_native_jit_disabled
            print(check_native_jit_disabled())
        """)
        env = os.environ.copy()
        env["TORCH_DISABLE_NATIVE_JIT"] = "1"
        result = _subprocess_lastline(script, env=env)
        self.assertEqual(result, "True")

    def test_unavailable_reason_present(self):
        """Known package -> _unavailable_reason returns None."""
        from torch._native.common_utils import _unavailable_reason

        self.assertIsNone(_unavailable_reason([("torch", "torch")]))

    def test_unavailable_reason_missing(self):
        """Nonexistent package -> _unavailable_reason returns a string."""
        from torch._native.common_utils import _unavailable_reason

        reason = _unavailable_reason([("nonexistent_pkg_xyz", "nonexistent_pkg_xyz")])
        self.assertIsNotNone(reason)
        self.assertIn("nonexistent_pkg_xyz", reason)

    def test_available_version(self):
        """_available_version returns a 3-tuple of ints for a clean version."""
        from torch._native.common_utils import _available_version

        # Use typing_extensions which always has a clean major.minor.patch version,
        # unlike torch which may have pre-release suffixes in dev builds.
        ver = _available_version("typing_extensions")
        self.assertIsInstance(ver, tuple)
        self.assertEqual(len(ver), 3)
        for v in ver:
            self.assertIsInstance(v, int)

    def test_registry_mechanics(self):
        """register_op_registerer + register_all_operators round-trips."""
        from torch._native.registry import (
            _RegisteredFns,
            register_all_operators,
            register_op_registerer,
        )

        sentinel = []
        fn = lambda: sentinel.append(True)

        original_len = len(_RegisteredFns)
        register_op_registerer(fn)
        self.assertEqual(len(_RegisteredFns), original_len + 1)

        register_all_operators()
        self.assertTrue(sentinel, "registered fn was never called")

        # cleanup
        _RegisteredFns.pop()

    def test_register_op_skips_when_runtime_unavailable(self):
        """register_op does not enqueue fn when runtime is unavailable."""
        from torch._native.registry import _RegisteredFns
        from torch._native import triton_utils, cutedsl_utils

        for mod, flag_name in [
            (triton_utils, "_TRITON_AVAILABLE"),
            (cutedsl_utils, "_CUTEDSL_AVAILABLE"),
        ]:
            original_len = len(_RegisteredFns)
            saved = getattr(mod, flag_name)
            try:
                setattr(mod, flag_name, False)
                mod.register_op(lambda: None)
                self.assertEqual(
                    len(_RegisteredFns),
                    original_len,
                    f"{mod.__name__}.register_op should not enqueue when runtime unavailable",
                )
            finally:
                setattr(mod, flag_name, saved)

    def test_register_op_skips_when_jit_disabled(self):
        """register_op does not enqueue fn when TORCH_DISABLE_NATIVE_JIT=1."""
        script = textwrap.dedent("""\
            from torch._native.registry import _RegisteredFns
            from torch._native import triton_utils, cutedsl_utils

            before = len(_RegisteredFns)
            triton_utils.register_op(lambda: None)
            cutedsl_utils.register_op(lambda: None)
            after = len(_RegisteredFns)
            print(before == after)
        """)
        env = os.environ.copy()
        env["TORCH_DISABLE_NATIVE_JIT"] = "1"
        result = _subprocess_lastline(script, env=env)
        self.assertEqual(result, "True")


if __name__ == "__main__":
    run_tests()
