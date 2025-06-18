#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>

namespace at::native {

template <typename scalar_t, typename index_t>
__global__ void segment_reduce_forward_kernel(
    ReductionType reduction,
    scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    bool is_initial_set,
    scalar_t initial_value,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis) {
  int64_t idx = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= (outer_offset * segment_count * inner_offset)) {
    return;
  }
  int64_t row_id = idx / inner_offset;
  int64_t lane_id = idx % inner_offset;   // lane_id is the inner_idx
  int64_t outer_idx = row_id / segment_count;
  int64_t dim_idx = row_id % segment_count;

  int64_t offset_idx = outer_idx * lengths_cumsum_stride_axis * (segment_count + 1) + dim_idx;
  index_t offset_start = lengths_cumsum_data[offset_idx];
  index_t offset_end = lengths_cumsum_data[offset_idx + 1];

  // ===== step2: apply reduction
  for (index_t j = offset_start; j < offset_end; ++j) {
    int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                         + j * data_stride_axis + lane_id;
    const auto data = values_data[data_index];
    // TODO: There is no need to branch with every element
    if (reduction == ReductionType::MAX) {
      initial_value =
          at::_isnan(data) ? data : std::max<scalar_t>(initial_value, data);
    } else if (
        reduction == ReductionType::MEAN ||
        reduction == ReductionType::SUM) {
      initial_value = initial_value + data;
    } else if (reduction == ReductionType::MIN) {
      initial_value =
          at::_isnan(data) ? data : std::min<scalar_t>(initial_value, data);
    } else if (
      reduction == ReductionType::PROD) {
      initial_value = initial_value * data;
    }
  }

  // ===== step3: finalize reduction
  int64_t lengths_idx = outer_idx * lengths_stride_axis * segment_count + dim_idx;
  CUDA_KERNEL_ASSERT(lengths_data[lengths_idx] >= 0);
  if (lengths_data[lengths_idx] == 0 && !is_initial_set &&
      reduction == ReductionType::MEAN) {
    initial_value = static_cast<scalar_t>(NAN);
  } else if (
      reduction == ReductionType::MEAN && lengths_data[lengths_idx] > 0 &&
      !at::_isnan(initial_value)) {
    initial_value = initial_value / lengths_data[lengths_idx];
  }
  int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                         + dim_idx * output_stride_axis + lane_id;
  output_data[output_index] = initial_value;
}

} // namespace at::native 