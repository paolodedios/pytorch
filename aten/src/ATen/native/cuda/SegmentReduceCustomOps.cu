#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>

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

template <typename scalar_t, typename index_t>
__global__ void post_sum_div_kernel(
    scalar_t* output_data,
    const index_t* lengths_data,
    const int64_t segment_count,
    bool is_initial_set,
    scalar_t initial) {
  CUDA_KERNEL_LOOP(index, segment_count) {
    CUDA_KERNEL_ASSERT(lengths_data[index] >= 0);
    if (lengths_data[index] == 0) {
      if (is_initial_set) {
        output_data[index] = initial;
      } else {
        output_data[index] = NAN;
      }
    } else if (!at::_isnan(output_data[index])) {
      output_data[index] = output_data[index] / lengths_data[index];
    }
  }
}

} // namespace at::native 