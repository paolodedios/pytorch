#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cumsum.h>
#endif

namespace at::native {

Tensor _segment_reduce_lengths_offsets_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& lengths_or_offsets,
  int64_t axis,
  const std::optional<Scalar>& initial,
  bool is_offsets_like) {
  // data and lengths_or_offsets should be contiguous from the call to .contiguous in segment_reduce_kernel
  TORCH_CHECK(data.is_contiguous());
  TORCH_CHECK(lengths_or_offsets.is_contiguous());
  axis = lengths_or_offsets.dim() - 1;
  int64_t segment_count = is_offsets_like ? lengths_or_offsets.size(axis) - 1 : lengths_or_offsets.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets.stride(axis);
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  auto offsets = lengths_or_offsets;
  auto lengths = lengths_or_offsets;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets = at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++) {
    outer_offset *= output.size(d);
  }
  for (int64_t d = axis + 1; d < output.dim(); d++) {
    inner_offset *= output.size(d);
  }

  if (output_shape.size() > 1) {
    // Use multi-dimensional kernel
    constexpr int threads_per_block = 256;
    int64_t num_blocks = (output.numel() + threads_per_block - 1) / threads_per_block;
    num_blocks = std::max(num_blocks, (int64_t)1);

    auto data_stride_axis = data.stride(axis);
    auto data_size_axis = data.size(axis);
    auto output_stride_axis = output.stride(axis);
    auto output_size_axis = output.size(axis);
    auto offsets_stride_axis = offsets.stride(axis);

    AT_DISPATCH_INDEX_TYPES(
        lengths_or_offsets.scalar_type(), "_segment_reduce_cuda_kernel1", ([&] {
          auto* offsets_data_ptr = offsets.const_data_ptr<index_t>();
          auto* lengths_data_ptr = lengths.const_data_ptr<index_t>();
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              data.scalar_type(),
              "segment_reduce_cuda",
              [&]() {
                auto* data_data_ptr = data.const_data_ptr<scalar_t>();
                auto* output_data_ptr = output.mutable_data_ptr<scalar_t>();

                // initialize starting value
                scalar_t initial_value = 0;
                if (initial.has_value()) {
                  initial_value = initial.value().to<scalar_t>();
                } else if (reduction == ReductionType::MAX) {
                  initial_value = -std::numeric_limits<scalar_t>::infinity();
                } else if (
                    reduction == ReductionType::MEAN ||
                    reduction == ReductionType::SUM) {
                  initial_value = 0;
                } else if (reduction == ReductionType::MIN) {
                  initial_value = std::numeric_limits<scalar_t>::infinity();
                } else if (reduction == ReductionType::PROD) {
                  initial_value = 1;
                }

                segment_reduce_forward_kernel<scalar_t>
                    <<<num_blocks,
                       threads_per_block,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        reduction,
                        output_data_ptr,
                        data_data_ptr,
                        lengths_data_ptr,
                        offsets_data_ptr,
                        segment_count,
                        lengths_stride_axis,
                        initial.has_value(),
                        initial_value,
                        outer_offset,
                        inner_offset,
                        data_stride_axis,
                        data_size_axis,
                        output_stride_axis,
                        output_size_axis,
                        offsets_stride_axis
                      );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        }));
  } else {
    // Use CUB for 1D case
    output = _segment_reduce_cub_cuda_kernel(
        reduction, data, lengths_or_offsets, offsets, lengths, 
        segment_count, initial, is_offsets_like);
  }

  return output;
}

Tensor _segment_reduce_lengths_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& lengths,
  int64_t axis,
  const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_cuda_kernel(
    reduction, data, lengths, axis, initial, /*is_offsets_like=*/false);
}

Tensor _segment_reduce_offsets_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& offsets,
  int64_t axis,
  const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_cuda_kernel(
    reduction, data, offsets, axis, initial, /*is_offsets_like=*/true);
}

} // namespace at::native 