// LossOps.metal
// Metal compute kernels replacing MPSGraph ops in LossOps.mm.
// Eliminates shape-dependent graph cache growth for smooth_l1 / huber loss.
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
// SmoothL1 / Huber Loss  (is_huber flag selects formula)
// ============================================================================

template <typename T>
kernel void smooth_huber_fwd_none(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  T t_beta = T(p.beta);
  auto sh_elem = [&](T d) -> T {
    T ad = select(-d, d, d >= T(0));
    T l_q = p.is_huber ? T(0.5) * d * d : T(0.5) * d * d / t_beta;
    T l_l = p.is_huber ? t_beta * (ad - T(0.5) * t_beta) : ad - T(0.5) * t_beta;
    return select(l_l, l_q, ad < t_beta);
  };
  uint base = gid * 4u;
  if (base + 4u <= p.N) {
    using T4 = vec<T, 4>;
    T4 i4 = reinterpret_cast<device const T4*>(input)[gid];
    T4 t4 = reinterpret_cast<device const T4*>(target)[gid];
    T4 res;
    res[0] = sh_elem(i4[0] - t4[0]);
    res[1] = sh_elem(i4[1] - t4[1]);
    res[2] = sh_elem(i4[2] - t4[2]);
    res[3] = sh_elem(i4[3] - t4[3]);
    reinterpret_cast<device T4*>(out)[gid] = res;
  } else {
    for (uint i = base; i < p.N; i++) {
      out[i] = sh_elem(input[i] - target[i]);
    }
  }
}

template <typename T>
kernel void smooth_huber_fwd_reduce(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  using T4 = vec<T, 4>;
  auto loss_elem = [&](float d) -> float {
    float ad = abs(d);
    return p.is_huber
        ? (ad < p.beta ? 0.5f * d * d : p.beta * (ad - 0.5f * p.beta))
        : (ad < p.beta ? 0.5f * d * d / p.beta : ad - 0.5f * p.beta);
  };
  uint aligned_N = (p.N / 4u) * 4u;
  uint base = gid * 4u;
  uint vec_stride = tpg * 4u;
  while (base + 4u <= aligned_N) {
    T4 i4 = reinterpret_cast<device const T4*>(input)[base / 4u];
    T4 t4 = reinterpret_cast<device const T4*>(target)[base / 4u];
    float4 d4 = float4(i4) - float4(t4);
    acc +=
        loss_elem(d4.x) + loss_elem(d4.y) + loss_elem(d4.z) + loss_elem(d4.w);
    base += vec_stride;
  }
  for (uint i = aligned_N + gid; i < p.N; i += tpg) {
    float d = float(input[i]) - float(target[i]);
    acc += loss_elem(d);
  }
  acc = c10::metal::threadgroup_sum<float>(smem, acc, lid, tgsz);
  if (lid == 0)
    partial[tgid] = acc * p.scale;
}

template <typename T>
kernel void smooth_huber_bwd(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* target [[buffer(2)]],
    device T* grad_in [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  auto dg_elem = [&](float d) -> float {
    float ad = abs(d);
    return p.is_huber ? (ad < p.beta ? d : p.beta * sign(d))
                      : (ad < p.beta ? d / p.beta : sign(d));
  };
  auto dg_vec4 = [&](float4 d4) -> float4 {
    return float4(
        dg_elem(d4[0]), dg_elem(d4[1]), dg_elem(d4[2]), dg_elem(d4[3]));
  };
  float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
  uint base = gid * 16u;
  if (base + 16u <= p.N) {
    uint t = gid * 4u;
    T4 i0 = reinterpret_cast<device const T4*>(input)[t + 0u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[t + 1u];
    T4 i2 = reinterpret_cast<device const T4*>(input)[t + 2u];
    T4 i3 = reinterpret_cast<device const T4*>(input)[t + 3u];
    T4 a0 = reinterpret_cast<device const T4*>(target)[t + 0u];
    T4 a1 = reinterpret_cast<device const T4*>(target)[t + 1u];
    T4 a2 = reinterpret_cast<device const T4*>(target)[t + 2u];
    T4 a3 = reinterpret_cast<device const T4*>(target)[t + 3u];
    float4 d0 = float4(i0) - float4(a0);
    float4 d1 = float4(i1) - float4(a1);
    float4 d2 = float4(i2) - float4(a2);
    float4 d3 = float4(i3) - float4(a3);
    float4 g0, g1, g2, g3;
    float4 dg0 = dg_vec4(d0), dg1 = dg_vec4(d1), dg2 = dg_vec4(d2),
           dg3 = dg_vec4(d3);
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 0u]) * dg0;
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 1u]) * dg1;
      g2 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 2u]) * dg2;
      g3 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 3u]) * dg3;
    } else {
      g0 = g_scalar * dg0;
      g1 = g_scalar * dg1;
      g2 = g_scalar * dg2;
      g3 = g_scalar * dg3;
    }
    reinterpret_cast<device T4*>(grad_in)[t + 0u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[t + 1u] = T4(g1);
    reinterpret_cast<device T4*>(grad_in)[t + 2u] = T4(g2);
    reinterpret_cast<device T4*>(grad_in)[t + 3u] = T4(g3);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float dg = dg_elem(d);
      float g = (p.reduction == 0) ? float(grad_out[i]) * dg : g_scalar * dg;
      grad_in[i] = T(g);
    }
  }
}

#define INSTANTIATE_SMOOTH_HUBER(T)                     \
  template [[host_name("smooth_huber_fwd_none_" #T)]]   \
  kernel void smooth_huber_fwd_none<T>(                 \
      device const T*,                                  \
      device const T*,                                  \
      device T*,                                        \
      constant SmoothHuberParams&,                      \
      uint);                                            \
  template [[host_name("smooth_huber_fwd_reduce_" #T)]] \
  kernel void smooth_huber_fwd_reduce<T>(               \
      device const T*,                                  \
      device const T*,                                  \
      device float*,                                    \
      constant SmoothHuberParams&,                      \
      uint,                                             \
      uint,                                             \
      uint,                                             \
      uint,                                             \
      uint);                                            \
  template [[host_name("smooth_huber_bwd_" #T)]]        \
  kernel void smooth_huber_bwd<T>(                      \
      device const T*,                                  \
      device const T*,                                  \
      device const T*,                                  \
      device T*,                                        \
      constant SmoothHuberParams&,                      \
      uint);

INSTANTIATE_SMOOTH_HUBER(float)
INSTANTIATE_SMOOTH_HUBER(half)
INSTANTIATE_SMOOTH_HUBER(bfloat)

// ============================================================================
// FUSED FWD + SAVED-GRAD KERNELS (kernel fusion for fwd+bwd autograd path)
// ============================================================================
// Forward writes loss AND a saved gradient factor; backward reads only
// grad_out + saved_dg, avoiding the second round-trip to load x, y.
// Wins only kick in at reduction=none + N >= 1M; the C++ dispatcher gates
// these via kFusionMinNumel and falls through to the standard custom kernel
// otherwise (which is already optimized for mean/sum via fwd_reduce).

template <typename T>
kernel void huber_fwd_sg(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    device T* saved_dg [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const float fbeta = p.beta;
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  if (base + VEC <= p.N) {
    T4 i0 = reinterpret_cast<device const T4*>(input)[gid * 2u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[gid * 2u + 1u];
    T4 t0 = reinterpret_cast<device const T4*>(target)[gid * 2u];
    T4 t1 = reinterpret_cast<device const T4*>(target)[gid * 2u + 1u];
    float4 d0 = float4(i0) - float4(t0);
    float4 d1 = float4(i1) - float4(t1);
    float4 dg0 = clamp(d0, float4(-fbeta), float4(fbeta));
    float4 dg1 = clamp(d1, float4(-fbeta), float4(fbeta));
    float4 ad0 = abs(d0), ad1 = abs(d1);
    float4 q0 = min(ad0, float4(fbeta)), q1 = min(ad1, float4(fbeta));
    float4 lin0 = ad0 - q0, lin1 = ad1 - q1;
    float4 l0 = 0.5f * q0 * q0 + float4(fbeta) * lin0;
    float4 l1 = 0.5f * q1 * q1 + float4(fbeta) * lin1;
    reinterpret_cast<device T4*>(out)[gid * 2u] = T4(l0);
    reinterpret_cast<device T4*>(out)[gid * 2u + 1u] = T4(l1);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u] = T4(dg0);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u + 1u] = T4(dg1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float dg = clamp(d, -fbeta, fbeta);
      float ad = abs(d);
      float q = min(ad, fbeta);
      float lin = ad - q;
      out[i] = T(0.5f * q * q + fbeta * lin);
      saved_dg[i] = T(dg);
    }
  }
}

template <typename T>
kernel void smooth_l1_fwd_sg(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    device T* saved_dg [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const float fbeta = p.beta;
  const float inv_beta = 1.0f / fbeta;
  const float inv_2beta = 0.5f * inv_beta;
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  if (base + VEC <= p.N) {
    T4 i0 = reinterpret_cast<device const T4*>(input)[gid * 2u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[gid * 2u + 1u];
    T4 t0 = reinterpret_cast<device const T4*>(target)[gid * 2u];
    T4 t1 = reinterpret_cast<device const T4*>(target)[gid * 2u + 1u];
    float4 d0 = float4(i0) - float4(t0);
    float4 d1 = float4(i1) - float4(t1);
    float4 dg0 = clamp(d0, float4(-fbeta), float4(fbeta)) * float4(inv_beta);
    float4 dg1 = clamp(d1, float4(-fbeta), float4(fbeta)) * float4(inv_beta);
    float4 ad0 = abs(d0), ad1 = abs(d1);
    float4 q0 = min(ad0, float4(fbeta)), q1 = min(ad1, float4(fbeta));
    float4 l0 = q0 * q0 * float4(inv_2beta) + (ad0 - q0);
    float4 l1 = q1 * q1 * float4(inv_2beta) + (ad1 - q1);
    reinterpret_cast<device T4*>(out)[gid * 2u] = T4(l0);
    reinterpret_cast<device T4*>(out)[gid * 2u + 1u] = T4(l1);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u] = T4(dg0);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u + 1u] = T4(dg1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float dg = clamp(d, -fbeta, fbeta) * inv_beta;
      float ad = abs(d);
      float q = min(ad, fbeta);
      out[i] = T(q * q * inv_2beta + (ad - q));
      saved_dg[i] = T(dg);
    }
  }
}

template <typename T>
kernel void huber_or_sl1_bwd_sg(
    device const T* grad_out [[buffer(0)]],
    device const T* saved_dg [[buffer(1)]],
    device T* grad_in [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
  if (base + VEC <= p.N) {
    T4 dg0 = reinterpret_cast<device const T4*>(saved_dg)[gid * 2u];
    T4 dg1 = reinterpret_cast<device const T4*>(saved_dg)[gid * 2u + 1u];
    float4 g0, g1;
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[gid * 2u]) *
          float4(dg0);
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[gid * 2u + 1u]) *
          float4(dg1);
    } else {
      g0 = g_scalar * float4(dg0);
      g1 = g_scalar * float4(dg1);
    }
    reinterpret_cast<device T4*>(grad_in)[gid * 2u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[gid * 2u + 1u] = T4(g1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float dg = float(saved_dg[i]);
      float gout = (p.reduction == 0) ? float(grad_out[i]) : g_scalar;
      grad_in[i] = T(gout * dg);
    }
  }
}

template [[host_name("huber_fwd_sg_half")]]
kernel void huber_fwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    device half*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_fwd_sg_bfloat")]]
kernel void huber_fwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    device bfloat*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_fwd_sg_float")]]
kernel void huber_fwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    device float*,
    constant SmoothHuberParams&,
    uint);

template [[host_name("smooth_l1_fwd_sg_half")]]
kernel void smooth_l1_fwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    device half*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("smooth_l1_fwd_sg_bfloat")]]
kernel void smooth_l1_fwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    device bfloat*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("smooth_l1_fwd_sg_float")]]
kernel void smooth_l1_fwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    device float*,
    constant SmoothHuberParams&,
    uint);

template [[host_name("huber_or_sl1_bwd_sg_half")]]
kernel void huber_or_sl1_bwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_or_sl1_bwd_sg_bfloat")]]
kernel void huber_or_sl1_bwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_or_sl1_bwd_sg_float")]]
kernel void huber_or_sl1_bwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    constant SmoothHuberParams&,
    uint);
