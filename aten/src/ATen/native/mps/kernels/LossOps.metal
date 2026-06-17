// LossOps.metal
// Metal compute kernels replacing MPSGraph ops in LossOps.mm.
// Eliminates shape-dependent graph cache growth for all loss operations.
//
// Reduction codes match ATen: 0=None, 1=Mean, 2=Sum
// Kernel naming convention:
//   <op>_fwd_none_<T>  : None reduction, writes typed T* output
//   <op>_fwd_reduce_<T>: Mean/Sum, writes float* partial sums (one per TG)
//   <op>_bwd_<T>       : backward, grad_out scalar (reduce) or per-elem (none)

#include <ATen/native/mps/kernels/LossOps.h>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Phase-2: merge per-threadgroup float partials into loss[0]
// Always dispatched with a single 256-thread threadgroup.
// ============================================================================

template <typename T>
kernel void loss_reduce_partials_typed(
    device const float* partial [[buffer(0)]],
    device T* loss [[buffer(1)]],
    constant uint32_t& nparts [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  for (uint i = lid; i < nparts; i += tgsz)
    acc += partial[i];
  acc = c10::metal::threadgroup_sum<float>(smem, acc, lid, tgsz);
  if (lid == 0)
    loss[0] = T(acc);
}

#define INST_REDUCE_PARTIALS(T)                      \
  template [[host_name("loss_reduce_partials_" #T)]] \
  kernel void loss_reduce_partials_typed<T>(         \
      device const float*, device T*, constant uint32_t&, uint, uint)
INST_REDUCE_PARTIALS(float);
INST_REDUCE_PARTIALS(half);
INST_REDUCE_PARTIALS(bfloat);

// ============================================================================
// NLL Loss 1-D  (input (N,C) log-probs, target (N,) int class indices)
// C++ handles final scale (Mean denominator) after phase-2 reduction.
// Caller must pre-zero grad_in before dispatching nll_loss_bwd.
// ============================================================================

template <typename T>
kernel void nll_loss_fwd_none(
    device const T* log_prob [[buffer(0)]], // (N, C)
    device const int* target [[buffer(1)]], // (N,)
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]], // (N,)
    constant NLLParams& p [[buffer(4)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint n = gid; n < p.N; n += tpg) {
    int t = target[n];
    if (t == p.ignore_index) {
      out[n] = T(0);
      continue;
    }
    if (t < 0 || t >= int(p.C)) {
      TORCH_REPORT_ERROR(
          error_buf,
          "nll_loss: Target ",
          t,
          " is out of bounds [0, ",
          int(p.C),
          ")");
      out[n] = T(0);
      continue;
    }
    float l = -float(log_prob[n * p.C + uint32_t(t)]);
    out[n] = T(p.has_weight ? l * float(weight[t]) : l);
  }
}

template <typename T>
kernel void nll_loss_fwd_reduce(
    device const T* log_prob [[buffer(0)]],
    device const int* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device float* partial [[buffer(3)]], // (n_tg,) loss sums
    device float* wpartial [[buffer(4)]], // (n_tg,) weight sums
    constant NLLParams& p [[buffer(5)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256], wsmem[256];
  float acc = 0.f, wacc = 0.f;
  for (uint n = gid; n < p.N; n += tpg) {
    int t = target[n];
    if (t == p.ignore_index)
      continue;
    if (t < 0 || t >= int(p.C)) {
      TORCH_REPORT_ERROR(
          error_buf,
          "nll_loss: Target ",
          t,
          " is out of bounds [0, ",
          int(p.C),
          ")");
      continue;
    }
    float w = p.has_weight ? float(weight[t]) : 1.f;
    acc += -float(log_prob[n * p.C + uint32_t(t)]) * w;
    wacc += w;
  }
  acc = c10::metal::threadgroup_sum<float>(smem, acc, lid, tgsz);
  wacc = c10::metal::threadgroup_sum<float>(wsmem, wacc, lid, tgsz);
  if (lid == 0) {
    partial[tgid] = acc;
    wpartial[tgid] = wacc;
  }
}

// Backward: writes -grad_out_scaled to grad_in[n, target[n]].
// Caller zeros grad_in before dispatch; each thread handles one n.
template <typename T>
kernel void nll_loss_bwd(
    device const T* grad_out [[buffer(0)]], // scalar (reduce) or (N,)
    device const int* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* grad_in [[buffer(3)]], // (N, C) pre-zeroed
    device const T* total_w [[buffer(4)]], // scalar weight sum (Mean)
    constant NLLParams& p [[buffer(5)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint n = gid; n < p.N; n += tpg) {
    int t = target[n];
    if (t == p.ignore_index)
      continue;
    if (t < 0 || t >= int(p.C)) {
      TORCH_REPORT_ERROR(
          error_buf,
          "nll_loss: Target ",
          t,
          " is out of bounds [0, ",
          int(p.C),
          ")");
      continue;
    }
    float w = p.has_weight ? float(weight[t]) : 1.f;
    float scale;
    if (p.reduction == 0) {
      scale = -float(grad_out[n]) * w;
    } else if (p.reduction == 1) {
      // Mean: divide by the summed weight. The N fallback only triggers in the
      // degenerate all-zero-weight case (forward yields NaN there); for the
      // standard unweighted path total_w is the non-ignored count, never 0
      // here.
      float denom = (float(total_w[0]) != 0.f) ? float(total_w[0]) : float(p.N);
      scale = -float(grad_out[0]) * w / denom;
    } else {
      scale = -float(grad_out[0]) * w;
    }
    grad_in[n * p.C + uint32_t(t)] = T(scale);
  }
}

#define INSTANTIATE_NLL(T)                          \
  template [[host_name("nll_loss_fwd_none_" #T)]]   \
  kernel void nll_loss_fwd_none<T>(                 \
      device const T*,                              \
      device const int*,                            \
      device const T*,                              \
      device T*,                                    \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint);                                        \
  template [[host_name("nll_loss_fwd_reduce_" #T)]] \
  kernel void nll_loss_fwd_reduce<T>(               \
      device const T*,                              \
      device const int*,                            \
      device const T*,                              \
      device float*,                                \
      device float*,                                \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint);                                        \
  template [[host_name("nll_loss_bwd_" #T)]]        \
  kernel void nll_loss_bwd<T>(                      \
      device const T*,                              \
      device const int*,                            \
      device const T*,                              \
      device T*,                                    \
      device const T*,                              \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint);

INSTANTIATE_NLL(float)
INSTANTIATE_NLL(half)
INSTANTIATE_NLL(bfloat)

// Phase-3 for NLL Mean reduction: divide typed output[0] by typed
// total_weight[0]. Dispatched with a single thread after the two
// encode_reduce_partials calls.
template <typename T>
kernel void nll_finalize_mean(
    device T* loss [[buffer(0)]],
    device const T* total_weight [[buffer(1)]]) {
  float tw = float(total_weight[0]);
  loss[0] = (tw == 0.f) ? T(NAN) : T(float(loss[0]) / tw);
}

#define INST_NLL_FINALIZE(T)                      \
  template [[host_name("nll_finalize_mean_" #T)]] \
  kernel void nll_finalize_mean<T>(device T*, device const T*)
INST_NLL_FINALIZE(float);
INST_NLL_FINALIZE(half);
INST_NLL_FINALIZE(bfloat);
