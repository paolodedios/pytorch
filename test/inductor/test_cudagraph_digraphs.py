# Owner(s): ["module: inductor"]

import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch._dynamo
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TEST_CUDA
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

__global__ void read_via_pointer_table_kernel(
    const float* const* pointer_table,
    float* out,
    int64_t n) {
  const float* in = pointer_table[0];
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] * 2.0f;
  }
}

void check_cuda_float_contiguous(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "expected a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "expected float32");
  TORCH_CHECK(x.is_contiguous(), "expected contiguous input");
}

torch::Tensor device_to_device_memcpy_1d(torch::Tensor x) {
  check_cuda_float_contiguous(x);

  auto out = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      out.data_ptr<float>(),
      x.const_data_ptr<float>(),
      x.nbytes(),
      cudaMemcpyDeviceToDevice,
      stream.stream()));
  return out;
}

torch::Tensor device_to_device_memcpy_2d(torch::Tensor x) {
  check_cuda_float_contiguous(x);
  TORCH_CHECK(x.dim() == 2, "expected a 2D input");

  auto out = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t row_bytes = static_cast<size_t>(x.size(1)) * sizeof(float);
  const size_t rows = static_cast<size_t>(x.size(0));
  C10_CUDA_CHECK(cudaMemcpy2DAsync(
      out.data_ptr<float>(),
      row_bytes,
      x.const_data_ptr<float>(),
      row_bytes,
      row_bytes,
      rows,
      cudaMemcpyDeviceToDevice,
      stream.stream()));
  return out;
}

torch::Tensor device_to_device_memcpy_3d(torch::Tensor x) {
  check_cuda_float_contiguous(x);
  TORCH_CHECK(x.dim() == 3, "expected a 3D input");

  auto out = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t width_bytes = static_cast<size_t>(x.size(2)) * sizeof(float);
  const size_t height = static_cast<size_t>(x.size(1));
  const size_t depth = static_cast<size_t>(x.size(0));

  cudaMemcpy3DParms params = {};
  params.srcPtr = make_cudaPitchedPtr(
      const_cast<float*>(x.const_data_ptr<float>()),
      width_bytes,
      width_bytes,
      height);
  params.dstPtr =
      make_cudaPitchedPtr(out.data_ptr<float>(), width_bytes, width_bytes, height);
  params.extent = make_cudaExtent(width_bytes, height, depth);
  params.kind = cudaMemcpyDeviceToDevice;
  C10_CUDA_CHECK(cudaMemcpy3DAsync(&params, stream.stream()));
  return out;
}

torch::Tensor device_host_device_roundtrip(torch::Tensor x) {
  check_cuda_float_contiguous(x);

  auto out = at::empty_like(x);
  auto host = at::empty(
      {x.numel()},
      at::TensorOptions()
          .dtype(x.scalar_type())
          .device(at::kCPU)
          .pinned_memory(true));

  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      host.data_ptr<float>(),
      x.const_data_ptr<float>(),
      x.nbytes(),
      cudaMemcpyDeviceToHost,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      out.data_ptr<float>(),
      host.const_data_ptr<float>(),
      x.nbytes(),
      cudaMemcpyHostToDevice,
      stream.stream()));

  at::getHostAllocator(at::kCUDA)->record_event(
      host.data_ptr<float>(), host.storage().data_ptr().get_context(), stream);
  return out;
}

torch::Tensor read_via_copied_pointer_table(torch::Tensor x) {
  check_cuda_float_contiguous(x);

  auto out = at::empty_like(x);
  auto device_table = at::empty({1}, x.options().dtype(at::kLong));
  auto host_table = at::empty(
      {1},
      at::TensorOptions()
          .dtype(at::kLong)
          .device(at::kCPU)
          .pinned_memory(true));

  host_table.data_ptr<int64_t>()[0] =
      reinterpret_cast<int64_t>(x.const_data_ptr<float>());

  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      device_table.data_ptr<int64_t>(),
      host_table.data_ptr<int64_t>(),
      sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream.stream()));

  at::getHostAllocator(at::kCUDA)->record_event(
      host_table.data_ptr<int64_t>(),
      host_table.storage().data_ptr().get_context(),
      stream);

  constexpr int threads = 256;
  const int64_t n = x.numel();
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  read_via_pointer_table_kernel<<<blocks, threads, 0, stream.stream()>>>(
      reinterpret_cast<const float* const*>(device_table.data_ptr<int64_t>()),
      out.data_ptr<float>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor device_to_device_memcpy_1d(torch::Tensor x);
torch::Tensor device_to_device_memcpy_2d(torch::Tensor x);
torch::Tensor device_to_device_memcpy_3d(torch::Tensor x);
torch::Tensor device_host_device_roundtrip(torch::Tensor x);
torch::Tensor read_via_copied_pointer_table(torch::Tensor x);
"""


_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        try:
            import ninja
        except ImportError as e:
            raise unittest.SkipTest("requires ninja") from e

        os.environ["PATH"] = f"{ninja.BIN_DIR}:{os.environ['PATH']}"
        build_dir = Path(
            tempfile.mkdtemp(prefix="torch_cudagraph_digraphs_pointer_table_ext_")
        )
        _EXTENSION = load_inline(
            name="cudagraph_digraphs_pointer_table_ext",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=[
                "device_to_device_memcpy_1d",
                "device_to_device_memcpy_2d",
                "device_to_device_memcpy_3d",
                "device_host_device_roundtrip",
                "read_via_copied_pointer_table",
            ],
            build_directory=str(build_dir),
            with_cuda=True,
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    return _EXTENSION


@torch.library.custom_op(
    "cudagraph_digraphs_repro::device_to_device_memcpy_1d",
    mutates_args=(),
)
def device_to_device_memcpy_1d(x: torch.Tensor) -> torch.Tensor:
    return extension().device_to_device_memcpy_1d(x)


@device_to_device_memcpy_1d.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "cudagraph_digraphs_repro::device_to_device_memcpy_2d",
    mutates_args=(),
)
def device_to_device_memcpy_2d(x: torch.Tensor) -> torch.Tensor:
    return extension().device_to_device_memcpy_2d(x)


@device_to_device_memcpy_2d.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "cudagraph_digraphs_repro::device_to_device_memcpy_3d",
    mutates_args=(),
)
def device_to_device_memcpy_3d(x: torch.Tensor) -> torch.Tensor:
    return extension().device_to_device_memcpy_3d(x)


@device_to_device_memcpy_3d.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "cudagraph_digraphs_repro::device_host_device_roundtrip",
    mutates_args=(),
)
def device_host_device_roundtrip(x: torch.Tensor) -> torch.Tensor:
    return extension().device_host_device_roundtrip(x)


@device_host_device_roundtrip.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "cudagraph_digraphs_repro::read_via_copied_pointer_table",
    mutates_args=(),
)
def read_via_copied_pointer_table(x: torch.Tensor) -> torch.Tensor:
    return extension().read_via_copied_pointer_table(x)


@read_via_copied_pointer_table.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def fn(x: torch.Tensor) -> torch.Tensor:
    return read_via_copied_pointer_table(x)


def d2d_1d_fn(x: torch.Tensor) -> torch.Tensor:
    return device_to_device_memcpy_1d(x)


def d2d_2d_fn(x: torch.Tensor) -> torch.Tensor:
    return device_to_device_memcpy_2d(x)


def d2d_3d_fn(x: torch.Tensor) -> torch.Tensor:
    return device_to_device_memcpy_3d(x)


def device_host_device_roundtrip_fn(x: torch.Tensor) -> torch.Tensor:
    return device_host_device_roundtrip(x)


def make_input(i: int, n: int) -> torch.Tensor:
    return (
        torch.arange(n, device="cuda", dtype=torch.float32) + 1000.0 * i
    ).contiguous()


def make_shaped_input(i: int, shape: tuple[int, ...]) -> torch.Tensor:
    return (
        torch.arange(
            torch.Size(shape).numel(),
            device="cuda",
            dtype=torch.float32,
        ).reshape(shape)
        + 1000.0 * i
    ).contiguous()


@torch._dynamo.disable
def make_pinned_cpu_like(x: torch.Tensor) -> torch.Tensor:
    # Dynamo currently cannot fake aten._pin_memory.default, so keep the pinned
    # allocation eager while compiling the device-to-host and host-to-device
    # copies that consume it.
    return torch.empty_like(x, device="cpu").pin_memory()


def pinned_cpu_roundtrip(x: torch.Tensor) -> torch.Tensor:
    x_cpu = make_pinned_cpu_like(x)
    x_cpu.copy_(x, non_blocking=True)
    x.copy_(x_cpu, non_blocking=True)
    return x


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@unittest.skipUnless(nvcc_exist(), "requires nvcc")
class CUDAGraphDigraphTests(TestCase):
    def test_first_capture_is_replayed(self):
        from torch._inductor import cudagraph_digraphs

        for preserve_memops in (False, True):
            with self.subTest(preserve_memops=preserve_memops):
                capture_count = 0

                def model(inputs):
                    nonlocal capture_count
                    capture_count += 1
                    return [inputs[0] + capture_count]

                x = torch.full((1024,), 3.0, device="cuda")
                with config.patch(
                    {
                        "triton.cudagraphs_preserve_memops": preserve_memops,
                        "triton.cudagraphs_separate_input_pool": False,
                    }
                ):
                    run, initial_outputs = cudagraph_digraphs.cudagraphify(
                        model,
                        [x],
                        device_index=0,
                        is_backward=False,
                        is_inference=True,
                    )
                    replay_input = torch.full((1024,), 7.0, device="cuda")
                    replay_outputs = run([replay_input])

                torch.cuda.synchronize()
                self.assertEqual(capture_count, 2)
                self.assertEqual(initial_outputs[0], x + 1)
                self.assertEqual(replay_outputs[0], replay_input + 1)

    def test_kernel_pointer_update_allows_different_capture_offsets(self):
        from torch._inductor import cudagraph_digraphs

        ptr = 0x1100
        other_ptr = 0x4200
        updates = cudagraph_digraphs._dynamic_pointer_updates_for_kernel(
            (ptr.to_bytes(8, sys.byteorder),),
            (other_ptr.to_bytes(8, sys.byteorder),),
            [cudagraph_digraphs.DynamicAllocation(0x1000, 0x1000, 2)],
            [0x1000],
            [cudagraph_digraphs.DynamicAllocation(0x4000, 0x1000, 7)],
            [0x4000],
        )

        self.assertEqual(
            updates,
            [
                cudagraph_digraphs.KernelParamPointerUpdate(
                    param_index=0,
                    param_byte_offset=0,
                    alloc_idx=2,
                    alloc_offset=0x100,
                )
            ],
        )

    def capture_pointwise_graph(
        self, size: int, *, two_kernels: bool
    ) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        tensors = [
            torch.ones(size, device="cuda"),
            torch.empty(size, device="cuda"),
            torch.empty(size, device="cuda"),
        ]
        x, intermediate, out = tensors
        with torch.cuda.stream(stream):
            torch.add(x, 1, out=intermediate)
            if two_kernels:
                torch.mul(intermediate, 2, out=out)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            torch.add(x, 1, out=intermediate)
            if two_kernels:
                torch.mul(intermediate, 2, out=out)
        return graph, tensors

    def test_graph_node_topological_order_validation(self):
        from torch._inductor import cudagraph_digraphs

        graph, keepalive = self.capture_pointwise_graph(4096, two_kernels=True)
        graph_handle = cudagraph_digraphs.cuda_runtime.cudaGraph_t(
            init_value=graph.raw_cuda_graph()
        )
        nodes = cudagraph_digraphs._graph_nodes(graph_handle)
        self.assertGreaterEqual(len(nodes), 2)
        cudagraph_digraphs._validate_graph_node_order_and_topology(nodes, nodes)

        reordered_nodes = list(nodes)
        reordered_nodes[0], reordered_nodes[1] = (
            reordered_nodes[1],
            reordered_nodes[0],
        )
        with self.assertRaisesRegex(RuntimeError, "same topological order"):
            cudagraph_digraphs._validate_graph_node_order_and_topology(
                nodes, reordered_nodes
            )
        self.assertTrue(keepalive)

    def test_graph_kernel_launch_validation(self):
        from torch._inductor import cudagraph_digraphs

        graph, keepalive = self.capture_pointwise_graph(256, two_kernels=False)
        other_graph, other_keepalive = self.capture_pointwise_graph(
            1 << 20, two_kernels=False
        )
        graph_handle = cudagraph_digraphs.cuda_runtime.cudaGraph_t(
            init_value=graph.raw_cuda_graph()
        )
        other_graph_handle = cudagraph_digraphs.cuda_runtime.cudaGraph_t(
            init_value=other_graph.raw_cuda_graph()
        )
        nodes = cudagraph_digraphs._graph_nodes(graph_handle)
        other_nodes = cudagraph_digraphs._graph_nodes(other_graph_handle)
        with self.assertRaisesRegex(RuntimeError, "kernel nodes differ"):
            cudagraph_digraphs._validate_graph_node_order_and_topology(
                nodes, other_nodes
            )
        self.assertTrue(keepalive)
        self.assertTrue(other_keepalive)

    def check_memcpy_case(
        self,
        fn,
        *,
        preserve_memops: bool,
        shape: tuple[int, ...] = (4096,),
        expected_error: str | None = None,
    ) -> None:
        extension()
        torch._dynamo.reset()
        counters.clear()
        torch.cuda.empty_cache()

        def run_case() -> None:
            with config.patch(
                {
                    "triton.cudagraphs": True,
                    "triton.cudagraphs_elide_input_output_copies": True,
                    "triton.cudagraph_trees": False,
                    "triton.cudagraphs_preserve_memops": preserve_memops,
                    "triton.cudagraph_or_error": True,
                }
            ):
                compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                keepalive: list[torch.Tensor] = []
                for i in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    x = make_shaped_input(i + 1, shape)
                    keepalive.append(x)
                    actual = compiled(x)
                    torch.cuda.synchronize()
                    self.assertEqual(actual, x)

                    gc.collect()
                    torch.cuda.empty_cache()

                self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

        if expected_error is not None:
            with self.assertRaisesRegex(
                (AssertionError, RuntimeError),
                expected_error,
            ):
                run_case()
        else:
            run_case()

    def test_device_to_device_1d_memcpy_node(self):
        for preserve_memops in (False, True):
            with self.subTest(preserve_memops=preserve_memops):
                self.check_memcpy_case(
                    d2d_1d_fn,
                    preserve_memops=preserve_memops,
                )

    def test_device_to_host_memcpy_node_is_rejected(self):
        for preserve_memops in (False, True):
            with self.subTest(preserve_memops=preserve_memops):
                self.check_memcpy_case(
                    device_host_device_roundtrip_fn,
                    preserve_memops=preserve_memops,
                    shape=(1024,),
                    expected_error="Unsupported CUDA graph memcpy kind",
                )

    def test_device_to_device_2d_memcpy_node_is_rejected(self):
        for preserve_memops in (False, True):
            with self.subTest(preserve_memops=preserve_memops):
                self.check_memcpy_case(
                    d2d_2d_fn,
                    preserve_memops=preserve_memops,
                    shape=(32, 64),
                    expected_error="Expected 1D CUDA graph memcpy node",
                )

    def test_device_to_device_3d_memcpy_node_is_rejected(self):
        for preserve_memops in (False, True):
            with self.subTest(preserve_memops=preserve_memops):
                self.check_memcpy_case(
                    d2d_3d_fn,
                    preserve_memops=preserve_memops,
                    shape=(4, 16, 32),
                    expected_error="Expected 1D CUDA graph memcpy node",
                )

    def check_copied_pointer_table(self, *, preserve_memops: bool) -> None:
        extension()
        torch._dynamo.reset()
        torch.cuda.empty_cache()

        with config.patch(
            {
                "triton.cudagraphs": True,
                "triton.cudagraphs_elide_input_output_copies": True,
                "triton.cudagraph_trees": False,
                "triton.cudagraphs_preserve_memops": preserve_memops,
            }
        ):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            keepalive: list[torch.Tensor] = []
            for i in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                x = make_input(i + 1, 4096)
                keepalive.append(x)
                expected = x * 2.0
                actual = compiled(x)
                torch.cuda.synchronize()
                self.assertEqual(actual, expected)

                gc.collect()
                torch.cuda.empty_cache()

    def test_copied_pointer_table_with_preserved_memcpy_node(self):
        self.check_copied_pointer_table(preserve_memops=True)

    def test_copied_pointer_table_with_memcpy_kernel(self):
        self.check_copied_pointer_table(preserve_memops=False)

    def test_pinned_cpu_roundtrip_copy_nodes_are_preserved(self):
        for preserve_memops in (False, True):
            with self.subTest(preserve_memops=preserve_memops):
                torch._dynamo.reset()
                counters.clear()
                torch.cuda.empty_cache()

                with config.patch(
                    {
                        "triton.cudagraphs": True,
                        "triton.cudagraphs_elide_input_output_copies": True,
                        "triton.cudagraph_trees": False,
                        "triton.cudagraphs_preserve_memops": preserve_memops,
                    }
                ):
                    compiled = torch.compile(
                        pinned_cpu_roundtrip, backend="inductor", fullgraph=False
                    )
                    for i in range(3):
                        torch.compiler.cudagraph_mark_step_begin()
                        x = make_input(i + 1, 64)
                        expected = x.clone()
                        actual = compiled(x)
                        torch.cuda.synchronize()
                        self.assertIs(actual, x)
                        self.assertEqual(actual, expected)

                        gc.collect()
                        torch.cuda.empty_cache()

                    self.assertGreater(counters["inductor"]["cudagraph_partitions"], 0)
                    self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

                    torch._dynamo.reset()
                    _, source_codes = run_and_get_code(
                        torch.compile(
                            pinned_cpu_roundtrip,
                            backend="inductor",
                            fullgraph=False,
                        ),
                        make_input(100, 64),
                    )
                    generated_code = "\n".join(source_codes)
                    self.assertIn("torch.ops.aten.copy.default", generated_code)
                    self.assertIn("torch.ops.aten.copy_.default", generated_code)
                    self.assertIn("Original ATen: [aten.copy", generated_code)
                    self.assertIn("aten.copy_", generated_code)


if __name__ == "__main__":
    run_tests("cuda")
