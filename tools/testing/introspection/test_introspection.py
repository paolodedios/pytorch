# Owner(s): ["module: ci"]
"""Tests for the test-introspection collector.

Fast tests cover the platform/descriptor logic. The collector smoke tests import a
real (small) test file in a subprocess and are marked slow.
"""

import pathlib
import tempfile

from tools.testing.introspection import collector, diff, platforms, where

from torch.testing._internal.common_utils import run_tests, slowTest, TestCase


INDUCTOR = "test/inductor/test_torchinductor.py"


class TestPlatforms(TestCase):
    def test_registry_get(self):
        self.assertEqual(platforms.get("linux-cpu").device_type, "cpu")
        with self.assertRaises(KeyError):
            platforms.get("does-not-exist")

    def test_cuda_caps_sm_derived(self):
        sm80 = platforms.get("linux-cuda-sm80")
        sm90 = platforms.get("linux-cuda-sm90")
        # FP8 needs SM89+, so it is off for SM80 and on for SM90.
        self.assertFalse(sm80.caps["PLATFORM_SUPPORTS_FP8"])
        self.assertTrue(sm90.caps["PLATFORM_SUPPORTS_FP8"])
        # Flash attention is SM80+.
        self.assertTrue(sm80.caps["PLATFORM_SUPPORTS_FLASH_ATTENTION"])

    def test_subprocess_env_hides_accelerators(self):
        env = platforms.get("linux-rocm").subprocess_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTORCH_TEST_WITH_ROCM"], "1")


class TestCollector(TestCase):
    @slowTest
    def test_device_gating(self):
        # GPU classes appear only on the cuda platform, not on cpu.
        cpu = collector.enumerate_tests(
            INDUCTOR, platforms.get_job("linux-cpu"), use_cache=False
        )
        cuda = collector.enumerate_tests(
            INDUCTOR, platforms.get_job("linux-cuda-sm80"), use_cache=False
        )
        self.assertNotIn("GPUTests", cpu)
        self.assertIn("GPUTests", cuda)
        self.assertIn("CpuTests", cpu)

    @slowTest
    def test_status_consistency(self):
        # ran union skipped must equal the enumerated set.
        job = platforms.get_job("linux-cpu")
        enum = collector.enumerate_tests(INDUCTOR, job, use_cache=False)
        enumerated = {f"{c}::{m}" for c, ms in enum.items() for m in ms}
        st = collector.status(INDUCTOR, job, use_cache=False)
        observed = set(st["ran"]) | {k for k, _ in st["skipped"]}
        self.assertEqual(enumerated, observed)


class TestDiff(TestCase):
    def test_is_test_py(self):
        self.assertTrue(diff._is_test_py("test/test_x.py"))
        self.assertTrue(diff._is_test_py("test/nn/test_pooling.py"))
        self.assertFalse(diff._is_test_py("test/helper.py"))
        self.assertFalse(diff._is_test_py("torch/x.py"))

    def test_is_broad(self):
        # Generation/selection surface + test infra are broad.
        self.assertTrue(diff._is_broad("torch/testing/_internal/common_utils.py"))
        self.assertTrue(diff._is_broad("tools/testing/discover_tests.py"))
        self.assertTrue(diff._is_broad("test/run_test.py"))
        self.assertTrue(diff._is_broad("test/conftest.py"))
        # Behavior-only source changes can't add/remove tests -> not broad.
        self.assertFalse(diff._is_broad("torch/csrc/foo.cpp"))
        self.assertFalse(diff._is_broad("torch/utils/flop_counter.py"))
        self.assertFalse(diff._is_broad("test/test_x.py"))
        self.assertFalse(diff._is_broad("test/nn/test_pooling.py"))
        # Non-.py data under test/ (xfail lists) marks xfails, not existence.
        self.assertFalse(
            diff._is_broad("test/inductor/pallas_expected_failures/CpuTests.test_foo")
        )

    def test_module_ids(self):
        self.assertEqual(
            diff._module_ids("test/inductor/test_x.py"), ("inductor.test_x", "test_x")
        )

    def test_scope_pulls_importers(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "test").mkdir()
            (root / "test" / "test_base.py").write_text("class A:\n    pass\n")
            (root / "test" / "test_dep.py").write_text("from test_base import A\n")
            (root / "test" / "test_other.py").write_text("import os\n")
            sel = ["test/test_base.py", "test/test_dep.py", "test/test_other.py"]
            graph = diff._build_import_graph(sel, root)
            aff = diff._scope(["test/test_base.py"], sel, graph)
            self.assertIn("test/test_dep.py", aff)  # synthetic dependent pulled in
            self.assertNotIn("test/test_other.py", aff)


class TestWhere(TestCase):
    def test_match(self):
        tid = "TestFooCUDA::test_bar_cuda_float32"
        self.assertTrue(where._match(tid, tid))  # full id
        self.assertTrue(where._match("test_bar_cuda_float32", tid))  # bare method
        self.assertTrue(where._match("test_bar", tid))  # substring
        self.assertFalse(where._match("test_baz", tid))


if __name__ == "__main__":
    run_tests()
