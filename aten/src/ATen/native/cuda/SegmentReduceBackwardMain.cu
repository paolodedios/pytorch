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
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

Tensor _segment_reduce_lengths_offsets_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like) {
  axis = lengths_or_offsets_contig.dim() - 1;
  int64_t segment_count = is_offsets_like ? lengths_or_offsets_contig.size(axis) - 1 : lengths_or_offsets_contig.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets_contig.stride(axis);
  auto grad_input = at::zeros_like(data_contig);

  auto offsets = lengths_or_offsets_contig;
  auto lengths = lengths_or_offsets_contig;
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
    outer_offset *= output_contig.size(d);
  }
  for (int64_t d = axis + 1; d < output_contig.dim(); d++) {
    inner_offset *= output_contig.size(d);
  }

  constexpr int threads_per_block = 256;
  int64_t num_blocks = (output_contig.numel() + threads_per_block - 1) / threads_per_block;
  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data_contig.stride(axis);
  auto data_size_axis = data_contig.size(axis);
  auto output_stride_axis = output_contig.stride(axis);
  auto output_size_axis = output_contig.size(axis);
  auto offsets_stride_axis = offsets.stride(axis);

  at::Tensor grad_input_reshaped = grad_input;
  at::Tensor grad_contig_reshaped = grad_contig;
  at::Tensor output_contig_reshaped = output_contig;
  at::Tensor data_contig_reshaped = data_contig;

  AT_DISPATCH_INDEX_TYPES(
      lengths_or_offsets_contig.scalar_type(), "_segment_reduce_backward_cuda_kernel1", ([&] {
        auto* offsets_data_ptr = offsets.const_data_ptr<index_t>();
        auto* lengths_data_ptr = lengths.const_data_ptr<index_t>();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            data_contig.scalar_type(),
            "segment_reduce_backward_cuda",
            [&]() {
              auto* grad_input_data_ptr = grad_input_reshaped.mutable_data_ptr<scalar_t>();
              auto* grad_data_ptr = grad_contig_reshaped.const_data_ptr<scalar_t>();
              auto* output_data_ptr = output_contig_reshaped.const_data_ptr<scalar_t>();
              auto* data_data_ptr = data_contig_reshaped.const_data_ptr<scalar_t>();

              // initialize starting value
              scalar_t initial_prod_value = 1;
              if (initial.has_value()) {
                initial_prod_value = initial.value().to<scalar_t>();
              }

              segment_reduce_backward_kernel<scalar_t>
                  <<<num_blocks,
                     threads_per_block,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      reduction,
                      grad_input_data_ptr,
                      grad_data_ptr,
                      output_data_ptr,
                      data_data_ptr,
                      lengths_data_ptr,
                      offsets_data_ptr,
                      segment_count,
                      lengths_stride_axis,
                      initial_prod_value,
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

  return grad_input;
}

Tensor _segment_reduce_lengths_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_cuda_kernel(
    grad_contig, output_contig, data_contig, reduction, lengths_contig, axis, initial, /*is_offsets_like=*/false);
}

Tensor _segment_reduce_offsets_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_cuda_kernel(
    grad_contig, output_contig, data_contig, reduction, offsets_contig, axis, initial, /*is_offsets_like=*/true);
}

} // namespace at::native 