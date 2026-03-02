#include <ATen/native/mps/kernels/Distance.h>
#include <metal_stdlib>
using namespace metal;

inline ulong row_start(ulong i, ulong n) {
  return n * i - i * (i + 1) / 2;
}

inline ulong2 pair_from_condensed_index(ulong k, ulong n) {
  float n2 = static_cast<float>(n) - 0.5f;
  ulong i = static_cast<ulong>(
      n2 - sqrt(n2 * n2 - 2.0f * static_cast<float>(k) - 1.0f));
  if (i >= n) {
    i = n - 1;
  }

  while (i > 0 && k < row_start(i, n)) {
    i--;
  }
  while ((i + 1) < n && k >= row_start(i + 1, n)) {
    i++;
  }

  ulong j = k - row_start(i, n) + i + 1;
  return ulong2(i, j);
}

inline float signf(float v) {
  return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
}

inline float forward_distance_update(
    float agg,
    float diff_abs,
    float p,
    PdistMode mode) {
  if (mode == PdistMode::MODE_ZERO) {
    return agg + min(ceil(diff_abs), 1.0f);
  }
  if (mode == PdistMode::MODE_ONE) {
    return agg + diff_abs;
  }
  if (mode == PdistMode::MODE_TWO) {
    return agg + diff_abs * diff_abs;
  }
  if (mode == PdistMode::MODE_INF) {
    return max(agg, diff_abs);
  }
  return agg + pow(diff_abs, p);
}

inline float forward_distance_finish(float agg, float p, PdistMode mode) {
  if (mode == PdistMode::MODE_TWO) {
    return sqrt(agg);
  }
  if (mode == PdistMode::MODE_GENERAL) {
    return pow(agg, 1.0f / p);
  }
  return agg;
}

inline float backward_value(
    float diff,
    float grad,
    float dist,
    float p,
    PdistMode mode) {
  if (dist == 0.0f) {
    return 0.0f;
  }

  float diff_abs = fabs(diff);
  float diff_sign = signf(diff);

  if (mode == PdistMode::MODE_ONE) {
    return grad * diff_sign;
  }
  if (mode == PdistMode::MODE_LT_TWO) {
    if (diff == 0.0f && p < 1.0f) {
      return 0.0f;
    }
    return diff_sign * pow(diff_abs, p - 1.0f) * grad / pow(dist, p - 1.0f);
  }
  if (mode == PdistMode::MODE_TWO) {
    return grad * diff / dist;
  }
  if (mode == PdistMode::MODE_INF) {
    return grad * diff_sign * (1.0f - min(1.0f, ceil(fabs(diff_abs - dist))));
  }
  return diff * pow(diff_abs, p - 2.0f) * grad / pow(dist, p - 1.0f);
}

template <typename T>
kernel void pdist_forward_kernel(
    device T* result [[buffer(0)]],
    constant T* self [[buffer(1)]],
    constant long& n [[buffer(2)]],
    constant long& m [[buffer(3)]],
    constant float& p [[buffer(4)]],
    constant int32_t& mode [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const PdistMode mode_enum = static_cast<PdistMode>(mode);
  ulong2 pair =
      pair_from_condensed_index(static_cast<ulong>(gid), static_cast<ulong>(n));
  ulong i = pair.x;
  ulong j = pair.y;

  float agg = 0.0f;
  for (ulong x = 0; x < static_cast<ulong>(m); ++x) {
    float a = static_cast<float>(self[i * static_cast<ulong>(m) + x]);
    float b = static_cast<float>(self[j * static_cast<ulong>(m) + x]);
    agg = forward_distance_update(agg, fabs(a - b), p, mode_enum);
  }

  result[gid] = static_cast<T>(forward_distance_finish(agg, p, mode_enum));
}

template <typename T>
kernel void pdist_backward_kernel(
    device T* buffer [[buffer(0)]],
    constant T* grad [[buffer(1)]],
    constant T* self [[buffer(2)]],
    constant T* dist [[buffer(3)]],
    constant long& grad_stride [[buffer(4)]],
    constant long& n [[buffer(5)]],
    constant long& m [[buffer(6)]],
    constant long& combs [[buffer(7)]],
    constant float& p [[buffer(8)]],
    constant int32_t& mode [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  const PdistMode mode_enum = static_cast<PdistMode>(mode);
  ulong k = static_cast<ulong>(gid) / static_cast<ulong>(m);
  ulong x = static_cast<ulong>(gid) % static_cast<ulong>(m);

  ulong2 pair = pair_from_condensed_index(k, static_cast<ulong>(n));
  ulong i = pair.x;
  ulong j = pair.y;
  ulong ib = j - i - 1;
  ulong jb = static_cast<ulong>(n - 2) - i;

  float grad_k = static_cast<float>(grad[k * static_cast<ulong>(grad_stride)]);
  float dist_k = static_cast<float>(dist[k]);
  float a = static_cast<float>(self[i * static_cast<ulong>(m) + x]);
  float b = static_cast<float>(self[j * static_cast<ulong>(m) + x]);

  float res = backward_value(a - b, grad_k, dist_k, p, mode_enum);

  ulong lhs = ((ib * static_cast<ulong>(n) + i) * static_cast<ulong>(m)) + x;
  ulong rhs = ((jb * static_cast<ulong>(n) + j) * static_cast<ulong>(m)) + x;
  buffer[lhs] = static_cast<T>(res);
  buffer[rhs] = static_cast<T>(-res);
}

#define REGISTER_PDIST_FORWARD_OP(DTYPE)                      \
  template [[host_name("pdist_forward_" #DTYPE)]] kernel void \
  pdist_forward_kernel<DTYPE>(                                \
      device DTYPE * result [[buffer(0)]],                    \
      constant DTYPE * self [[buffer(1)]],                    \
      constant long& n [[buffer(2)]],                         \
      constant long& m [[buffer(3)]],                         \
      constant float& p [[buffer(4)]],                        \
      constant int32_t& mode [[buffer(5)]],                   \
      uint gid [[thread_position_in_grid]]);

#define REGISTER_PDIST_BACKWARD_OP(DTYPE)                      \
  template [[host_name("pdist_backward_" #DTYPE)]] kernel void \
  pdist_backward_kernel<DTYPE>(                                \
      device DTYPE * buffer [[buffer(0)]],                     \
      constant DTYPE * grad [[buffer(1)]],                     \
      constant DTYPE * self [[buffer(2)]],                     \
      constant DTYPE * dist [[buffer(3)]],                     \
      constant long& grad_stride [[buffer(4)]],                \
      constant long& n [[buffer(5)]],                          \
      constant long& m [[buffer(6)]],                          \
      constant long& combs [[buffer(7)]],                      \
      constant float& p [[buffer(8)]],                         \
      constant int32_t& mode [[buffer(9)]],                    \
      uint gid [[thread_position_in_grid]]);

REGISTER_PDIST_FORWARD_OP(float);
REGISTER_PDIST_FORWARD_OP(half);
REGISTER_PDIST_FORWARD_OP(bfloat);

REGISTER_PDIST_BACKWARD_OP(float);
REGISTER_PDIST_BACKWARD_OP(half);
REGISTER_PDIST_BACKWARD_OP(bfloat);
