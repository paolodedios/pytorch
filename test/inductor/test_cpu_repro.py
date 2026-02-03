# Owner(s): ["oncall: cpu inductor"]
import contextlib
import copy
import functools
import itertools
import math
import os
import platform
import sys
import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from torch import nn
from torch._C import FileCheck
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config, cpu_vec_isa, metrics, test_operators
from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.exc import InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import timed
from torch._prims_common import is_float_dtype
from torch.autograd.functional import vjp
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    parametrize,
    slowTest,
    TEST_MKL,
    xfailIfS390X,
)
from torch.utils._python_dispatch import TorchDispatchMode


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


vec_dtypes = test_torchinductor.vec_dtypes
_lowp_fp_dtypes = (
    torch.bfloat16,
    torch.float16,
)
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase
aten = torch.ops.aten
check_model = test_torchinductor.check_model

requires_vectorization = unittest.skipUnless(
    cpu_vec_isa.valid_vec_isa_list() and os.getenv("ATEN_CPU_CAPABILITY") != "default",
    "Does not support vectorization",
)


def _can_check_vec_metrics():
    return (
        cpu_vec_isa.valid_vec_isa_list()
        and os.getenv("ATEN_CPU_CAPABILITY") != "default"
        and config.cpp.simdlen != 1
    )


def check_metrics_vec_kernel_count(num_expected_vec_kernels):
    if _can_check_vec_metrics():
        assert metrics.generated_cpp_vec_kernel_count == num_expected_vec_kernels


def simd_lengths_to_test():
    """Returns a minimal list of simd lengths to cover common cases"""
    simdlens = [None, 1]
    valid_isa_list = cpu_vec_isa.valid_vec_isa_list()
    if valid_isa_list:
        simdlens.append(valid_isa_list[0].bit_width())
    return simdlens


@contextlib.contextmanager
def set_num_threads(num_threads):
    orig_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    yield
    torch.set_num_threads(orig_num_threads)


class LstmModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        bidirectional=False,
        batch_first=False,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


@instantiate_parametrized_tests
class CPUReproTests(TestCase):
    common = check_model

    def test_torch_linalg_qr_tuple_slice(self):
        def fn(x):
            return torch.linalg.qr(x)[:1]

        x = torch.randn(4, 4)
        compiled = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled(x)

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 1)
        torch.testing.assert_close(actual[0], expected[0])

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_stride_constraints(self):
        for fmt in [torch.contiguous_format, torch.channels_last]:
            # TorchDispatch doesn't work in our cuda invocation for some reason
            m = torch.nn.Conv2d(5, 6, [3, 3])

            def fn(inp, weight):
                return (
                    F.conv2d(
                        inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                    ),
                )

            inp = torch.randn([2, 5, 16, 16])
            inps = [inp, m.weight.to(memory_format=fmt)]
            fn_fx = make_fx(fn)(*inps)
            fn_compiled = compile_fx_inner(fn_fx, inps)
            test_self = self
            conv_seen = False

            class RecordFunctions(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs if kwargs else {}
                    if func == torch.ops.aten.convolution.default:
                        # For CPU and mkldnn enable, we always using channels last
                        nonlocal fmt
                        if (
                            torch.backends.mkldnn.enabled
                            and torch.backends.mkldnn.is_available()
                        ):
                            fmt = torch.channels_last
                        test_self.assertTrue(args[0].is_contiguous(memory_format=fmt))
                        test_self.assertTrue(args[1].is_contiguous(memory_format=fmt))
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            with RecordFunctions():
                fn_compiled(inps)

            self.assertTrue(conv_seen)