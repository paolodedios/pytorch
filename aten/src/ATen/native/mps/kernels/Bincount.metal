#include <metal_stdlib>
using namespace metal;

// Atomically increments the count for each input element's bin. The accumulator
// is uint32 because (a) Metal atomics on integer types are 32-bit on all
// supported Apple GPUs, and (b) PyTorch tensors are bounded in size such that
// numel fits comfortably in uint32. Bins are bounded by nbins; the host
// guarantees indices[tid] is in [0, nbins).
template <typename IDX_T>
kernel void bincount_unweighted(
    constant IDX_T* indices [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant ulong& numel [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  long bin = long(indices[tid]);
  atomic_fetch_add_explicit(&counts[bin], 1u, memory_order_relaxed);
}

// Per-element float-weighted bincount. The accumulator is atomic<float>; Metal
// supports atomic_fetch_add_explicit on atomic<float> on Apple Silicon
// (Metal 3 / macOS 13+).
template <typename IDX_T>
kernel void bincount_weighted_float(
    constant IDX_T* indices [[buffer(0)]],
    constant float* weights [[buffer(1)]],
    device atomic_float* output [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  long bin = long(indices[tid]);
  atomic_fetch_add_explicit(&output[bin], weights[tid], memory_order_relaxed);
}

// Per-element int32-weighted bincount.
template <typename IDX_T>
kernel void bincount_weighted_int(
    constant IDX_T* indices [[buffer(0)]],
    constant int* weights [[buffer(1)]],
    device atomic_int* output [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  long bin = long(indices[tid]);
  atomic_fetch_add_explicit(&output[bin], weights[tid], memory_order_relaxed);
}

// Widen uint32 counts into int64 (the canonical output dtype for unweighted
// bincount). Run after bincount_unweighted has finished.
kernel void bincount_widen_uint_to_long(
    constant uint* counts [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant ulong& nbins [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= nbins) {
    return;
  }
  output[tid] = long(counts[tid]);
}

#define REGISTER_BINCOUNT_FOR_IDX(IDX_T, IDX_NAME)                         \
  template [[host_name("bincount_unweighted_" #IDX_NAME)]] kernel void     \
  bincount_unweighted<IDX_T>(                                              \
      constant IDX_T * indices [[buffer(0)]],                              \
      device atomic_uint * counts [[buffer(1)]],                           \
      constant ulong & numel [[buffer(2)]],                                \
      uint tid [[thread_position_in_grid]]);                               \
  template [[host_name("bincount_weighted_float_" #IDX_NAME)]] kernel void \
  bincount_weighted_float<IDX_T>(                                          \
      constant IDX_T * indices [[buffer(0)]],                              \
      constant float* weights [[buffer(1)]],                               \
      device atomic_float* output [[buffer(2)]],                           \
      constant ulong& numel [[buffer(3)]],                                 \
      uint tid [[thread_position_in_grid]]);                               \
  template [[host_name("bincount_weighted_int_" #IDX_NAME)]] kernel void   \
  bincount_weighted_int<IDX_T>(                                            \
      constant IDX_T * indices [[buffer(0)]],                              \
      constant int* weights [[buffer(1)]],                                 \
      device atomic_int* output [[buffer(2)]],                             \
      constant ulong& numel [[buffer(3)]],                                 \
      uint tid [[thread_position_in_grid]]);

REGISTER_BINCOUNT_FOR_IDX(char, char)
REGISTER_BINCOUNT_FOR_IDX(short, short)
REGISTER_BINCOUNT_FOR_IDX(int, int)
REGISTER_BINCOUNT_FOR_IDX(long, long)
REGISTER_BINCOUNT_FOR_IDX(uchar, uchar)
