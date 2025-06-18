#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>

namespace at::native {

template <typename scalar_t, typename index_t>
__global__ void segment_reduce_backward_kernel(
    ReductionType reduction,
    scalar_t* grad_input_data,
    const scalar_t* grad_data,
    const scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    scalar_t initial_prod_value,
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
  int64_t lane_id = idx % inner_offset;  // lane_id is the inner_idx
  int64_t outer_idx = row_id / segment_count;
  int64_t dim_idx = row_id % segment_count;

  int64_t lengths_idx = outer_idx * lengths_stride_axis * segment_count + dim_idx;
  auto segment_length = lengths_data[lengths_idx];
  if (segment_length == 0) {
    return;
  }

  int64_t offset_idx = outer_idx * lengths_cumsum_stride_axis * (segment_count + 1) + dim_idx;
  index_t offset_start = lengths_cumsum_data[offset_idx];
  index_t offset_end = lengths_cumsum_data[offset_idx + 1];

  int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                         + dim_idx * output_stride_axis + lane_id;

  if (reduction == ReductionType::MAX ||
      reduction == ReductionType::MIN) {
    int64_t counter = 0;
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                           + j * data_stride_axis + lane_id;
      if (at::_isnan(values_data[data_index]) ||
          values_data[data_index] == output_data[output_index]) {
        grad_input_data[data_index] = grad_data[output_index];
        counter++;
      }
    }
    // Average gradient based on number of maximum elements in the
    // segment
    if (counter < 2) {
      return;
    }
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                           + j * data_stride_axis + lane_id;
      if (grad_input_data[data_index] > 0) {
        grad_input_data[data_index] =
            grad_input_data[data_index] / counter;
      }
    }
  } else if (reduction == ReductionType::MEAN) {
    auto grad_val = grad_data[output_index] / segment_length;
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                           + j * data_stride_axis + lane_id;
      grad_input_data[data_index] = grad_val;
    }
  } else if (reduction == ReductionType::SUM) {
    const auto& grad_val = grad_data[output_index];
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                           + j * data_stride_axis + lane_id;
      grad_input_data[data_index] = grad_val;
    }
  } else if (reduction == ReductionType::PROD) {
    const auto& grad_val = grad_data[output_index] * output_data[output_index];
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                           + j * data_stride_axis + lane_id;
      if (at::_isnan(values_data[data_index]) ||
          values_data[data_index] == 0) {
        // explicitly compute exclusive prod
        scalar_t exclusive_prod = initial_prod_value;
        int64_t prod_idx;
        for (int64_t k = offset_start; k < offset_end; ++k) {
          if (k != j) {
            prod_idx = outer_idx * data_stride_axis * data_size_axis
                       + k * data_stride_axis + lane_id;
            exclusive_prod *= values_data[prod_idx];
          }
        }
        grad_input_data[data_index] = grad_data[output_index] * exclusive_prod;
      } else {
        grad_input_data[data_index] = grad_val / values_data[data_index];
      }
    }
  }
}

} // namespace at::native 