#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/cub.cuh>

namespace at::native {

namespace {
struct CustomMax {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    if (at::_isnan(a)) {
      return a;
    } else if (at::_isnan(b)) {
      return b;
    }
    return std::max<OutputT>(a, b);
  }
};

struct CustomSum {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    return a + b;
  }
};

struct CustomProd {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    return a * b;
  }
};

struct CustomMin {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    if (at::_isnan(a)) {
      return a;
    } else if (at::_isnan(b)) {
      return b;
    }
    return std::min<OutputT>(a, b);
  }
};
} // namespace

Tensor _segment_reduce_cub_cuda_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths_or_offsets,
    const Tensor& offsets,
    const Tensor& lengths,
    int64_t segment_count,
    const std::optional<Scalar>& initial,
    bool is_offsets_like) {
  
  auto output_shape = data.sizes().vec();
  output_shape[lengths_or_offsets.dim() - 1] = segment_count;
  auto output = at::empty(output_shape, data.options());

  constexpr int threads_per_block = 256;
  int64_t num_blocks = (segment_count + threads_per_block - 1) / threads_per_block;
  num_blocks = std::max(num_blocks, (int64_t)1);

  AT_DISPATCH_INDEX_TYPES(
      lengths_or_offsets.scalar_type(), "_segment_reduce_cub_cuda_kernel", ([&] {
        auto* offsets_data_ptr = offsets.const_data_ptr<index_t>();
        auto* lengths_data_ptr = lengths.const_data_ptr<index_t>();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            data.scalar_type(),
            "segment_reduce_cub_cuda",
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

              if (reduction == ReductionType::MAX) {
                CustomMax max_op{};
                CUB_WRAPPER(
                    cub::DeviceSegmentedReduce::Reduce,
                    data_data_ptr,
                    output_data_ptr,
                    segment_count,
                    offsets_data_ptr,
                    offsets_data_ptr + 1,
                    max_op,
                    initial_value,
                    at::cuda::getCurrentCUDAStream());
              } else if (reduction == ReductionType::MEAN) {
                CustomSum sum_op{};
                CUB_WRAPPER(
                    cub::DeviceSegmentedReduce::Reduce,
                    data_data_ptr,
                    output_data_ptr,
                    segment_count,
                    offsets_data_ptr,
                    offsets_data_ptr + 1,
                    sum_op,
                    initial_value,
                    at::cuda::getCurrentCUDAStream());

                post_sum_div_kernel<scalar_t>
                    <<<num_blocks,
                       threads_per_block,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        output_data_ptr,
                        lengths_data_ptr,
                        segment_count,
                        initial.has_value(),
                        initial_value);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else if (reduction == ReductionType::MIN) {
                CustomMin min_op{};
                CUB_WRAPPER(
                    cub::DeviceSegmentedReduce::Reduce,
                    data_data_ptr,
                    output_data_ptr,
                    segment_count,
                    offsets_data_ptr,
                    offsets_data_ptr + 1,
                    min_op,
                    initial_value,
                    at::cuda::getCurrentCUDAStream());
              } else if (reduction == ReductionType::SUM) {
                CustomSum sum_op{};
                CUB_WRAPPER(
                    cub::DeviceSegmentedReduce::Reduce,
                    data_data_ptr,
                    output_data_ptr,
                    segment_count,
                    offsets_data_ptr,
                    offsets_data_ptr + 1,
                    sum_op,
                    initial_value,
                    at::cuda::getCurrentCUDAStream());
              } else if (reduction == ReductionType::PROD) {
                CustomProd prod_op{};
                CUB_WRAPPER(
                    cub::DeviceSegmentedReduce::Reduce,
                    data_data_ptr,
                    output_data_ptr,
                    segment_count,
                    offsets_data_ptr,
                    offsets_data_ptr + 1,
                    prod_op,
                    initial_value,
                    at::cuda::getCurrentCUDAStream());
              }
            });
      }));

  return output;
}

} // namespace at::native 