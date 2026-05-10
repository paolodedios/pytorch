// LossOps.metal
// Metal compute kernels replacing MPSGraph ops in LossOps.mm.
// Eliminates shape-dependent graph cache growth for all loss operations.
//
// Reduction codes match ATen: 0=None, 1=Mean, 2=Sum
// Kernel naming convention:
//   <op>_fwd_none_<T>  : None reduction, writes typed T* output
//   <op>_fwd_reduce_<T>: Mean/Sum, writes float* partial sums (one per TG)
//   <op>_bwd_<T>       : backward, grad_out scalar (reduce) or per-elem (none)

#include <metal_stdlib>
using namespace metal;

// ──────────────────────────────────────────────────────────────────────────
// Param structs  (binary layout must match C++ structs in LossOps.mm)
// ──────────────────────────────────────────────────────────────────────────

struct PointwiseLossParams {
    uint32_t N;
    float    scale;      // 1/N (Mean) or 1.0 (Sum/None)
    uint32_t reduction;
};

struct SmoothHuberParams {
    uint32_t N;
    float    scale;
    uint32_t reduction;
    float    beta;
    uint32_t is_huber;   // 0=SmoothL1, 1=HuberLoss
};

struct BCEParams {
    uint32_t N;
    float    scale;
    uint32_t reduction;
    uint32_t has_weight;
};

struct NLLParams {
    uint32_t N;
    uint32_t C;
    int32_t  ignore_index;
    uint32_t reduction;
    uint32_t has_weight;
};

// ──────────────────────────────────────────────────────────────────────────
// Threadgroup tree-reduce helper
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
inline T tg_sum(T val, threadgroup T* smem, uint lid, uint tgsz) {
    smem[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return smem[0];
}

// ──────────────────────────────────────────────────────────────────────────
// Phase-2: merge per-threadgroup float partials into loss[0]
// Always dispatched with a single 256-thread threadgroup.
// ──────────────────────────────────────────────────────────────────────────

kernel void loss_reduce_partials(
    device const float* partial [[buffer(0)]],
    device       float* loss    [[buffer(1)]],
    constant uint32_t&  nparts  [[buffer(2)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float smem[256];
    float acc = 0.f;
    for (uint i = lid; i < nparts; i += tgsz) acc += partial[i];
    acc = tg_sum(acc, smem, lid, tgsz);
    if (lid == 0) loss[0] = acc;
}

// ============================================================================
// MSE Loss
// ============================================================================

template <typename T>
kernel void mse_loss_fwd_none(
    device const T*  input   [[buffer(0)]],
    device const T*  target  [[buffer(1)]],
    device       T*  out     [[buffer(2)]],
    constant uint32_t& N     [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    for (uint i = gid; i < N; i += tpg) {
        float d = float(input[i]) - float(target[i]);
        out[i] = T(d * d);
    }
}

template <typename T>
kernel void mse_loss_fwd_reduce(
    device const T*     input   [[buffer(0)]],
    device const T*     target  [[buffer(1)]],
    device       float* partial [[buffer(2)]],  // (n_tg,) float
    constant PointwiseLossParams& p [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint tpg   [[threads_per_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tgsz  [[threads_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]])
{
    threadgroup float smem[256];
    float acc = 0.f;
    for (uint i = gid; i < p.N; i += tpg) {
        float d = float(input[i]) - float(target[i]);
        acc += d * d;
    }
    acc = tg_sum(acc, smem, lid, tgsz);
    if (lid == 0) partial[tgid] = acc * p.scale;
}

template <typename T>
kernel void mse_loss_bwd(
    device const T*  grad_out [[buffer(0)]],  // scalar (reduce) or size-N (none)
    device const T*  input    [[buffer(1)]],
    device const T*  target   [[buffer(2)]],
    device       T*  grad_in  [[buffer(3)]],
    constant PointwiseLossParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale * 2.f : 0.f;
    for (uint i = gid; i < p.N; i += tpg) {
        float d = float(input[i]) - float(target[i]);
        float g = (p.reduction == 0) ? float(grad_out[i]) * 2.f * d
                                     : g_scalar * d;
        grad_in[i] = T(g);
    }
}

#define INSTANTIATE_MSE(T)                                                  \
  template [[host_name("mse_loss_fwd_none_" #T)]]                          \
  kernel void mse_loss_fwd_none<T>(                                        \
      device const T*, device const T*, device T*,                        \
      constant uint32_t&, uint, uint);                                     \
  template [[host_name("mse_loss_fwd_reduce_" #T)]]                        \
  kernel void mse_loss_fwd_reduce<T>(                                      \
      device const T*, device const T*, device float*,                    \
      constant PointwiseLossParams&, uint, uint, uint, uint, uint);        \
  template [[host_name("mse_loss_bwd_" #T)]]                               \
  kernel void mse_loss_bwd<T>(                                             \
      device const T*, device const T*, device const T*, device T*,       \
      constant PointwiseLossParams&, uint, uint);

INSTANTIATE_MSE(float)
INSTANTIATE_MSE(half)
INSTANTIATE_MSE(bfloat)

// ============================================================================
// SmoothL1 / Huber Loss  (is_huber flag selects formula)
// ============================================================================

template <typename T>
kernel void smooth_huber_fwd_none(
    device const T*  input  [[buffer(0)]],
    device const T*  target [[buffer(1)]],
    device       T*  out    [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    for (uint i = gid; i < p.N; i += tpg) {
        float d  = float(input[i]) - float(target[i]);
        float ad = abs(d);
        float l  = p.is_huber
            ? (ad < p.beta ? 0.5f * d * d          : p.beta * (ad - 0.5f * p.beta))
            : (ad < p.beta ? 0.5f * d * d / p.beta : ad - 0.5f * p.beta);
        out[i] = T(l);
    }
}

template <typename T>
kernel void smooth_huber_fwd_reduce(
    device const T*     input   [[buffer(0)]],
    device const T*     target  [[buffer(1)]],
    device       float* partial [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint tpg   [[threads_per_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tgsz  [[threads_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]])
{
    threadgroup float smem[256];
    float acc = 0.f;
    for (uint i = gid; i < p.N; i += tpg) {
        float d  = float(input[i]) - float(target[i]);
        float ad = abs(d);
        acc += p.is_huber
            ? (ad < p.beta ? 0.5f * d * d          : p.beta * (ad - 0.5f * p.beta))
            : (ad < p.beta ? 0.5f * d * d / p.beta : ad - 0.5f * p.beta);
    }
    acc = tg_sum(acc, smem, lid, tgsz);
    if (lid == 0) partial[tgid] = acc * p.scale;
}

template <typename T>
kernel void smooth_huber_bwd(
    device const T*  grad_out [[buffer(0)]],
    device const T*  input    [[buffer(1)]],
    device const T*  target   [[buffer(2)]],
    device       T*  grad_in  [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
    for (uint i = gid; i < p.N; i += tpg) {
        float d  = float(input[i]) - float(target[i]);
        float ad = abs(d);
        float dg = p.is_huber
            ? (ad < p.beta ? d          : p.beta * sign(d))
            : (ad < p.beta ? d / p.beta : sign(d));
        float g = (p.reduction == 0) ? float(grad_out[i]) * dg : g_scalar * dg;
        grad_in[i] = T(g);
    }
}

#define INSTANTIATE_SMOOTH_HUBER(T)                                          \
  template [[host_name("smooth_huber_fwd_none_" #T)]]                       \
  kernel void smooth_huber_fwd_none<T>(                                     \
      device const T*, device const T*, device T*,                         \
      constant SmoothHuberParams&, uint, uint);                             \
  template [[host_name("smooth_huber_fwd_reduce_" #T)]]                     \
  kernel void smooth_huber_fwd_reduce<T>(                                   \
      device const T*, device const T*, device float*,                     \
      constant SmoothHuberParams&, uint, uint, uint, uint, uint);           \
  template [[host_name("smooth_huber_bwd_" #T)]]                            \
  kernel void smooth_huber_bwd<T>(                                          \
      device const T*, device const T*, device const T*, device T*,        \
      constant SmoothHuberParams&, uint, uint);

INSTANTIATE_SMOOTH_HUBER(float)
INSTANTIATE_SMOOTH_HUBER(half)
INSTANTIATE_SMOOTH_HUBER(bfloat)

// ============================================================================
// BCE Loss  (binary cross-entropy; input clamped to (eps, 1-eps))
// ============================================================================

template <typename T>
kernel void bce_loss_fwd_none(
    device const T*  input   [[buffer(0)]],
    device const T*  target  [[buffer(1)]],
    device const T*  weight  [[buffer(2)]],
    device       T*  out     [[buffer(3)]],
    constant BCEParams& p    [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    for (uint i = gid; i < p.N; i += tpg) {
        float x = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
        float y = float(target[i]);
        float l = -y * log(x) - (1.f - y) * log(1.f - x);
        out[i] = T(p.has_weight ? l * float(weight[i]) : l);
    }
}

template <typename T>
kernel void bce_loss_fwd_reduce(
    device const T*     input   [[buffer(0)]],
    device const T*     target  [[buffer(1)]],
    device const T*     weight  [[buffer(2)]],
    device       float* partial [[buffer(3)]],
    constant BCEParams& p       [[buffer(4)]],
    uint gid   [[thread_position_in_grid]],
    uint tpg   [[threads_per_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tgsz  [[threads_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]])
{
    threadgroup float smem[256];
    float acc = 0.f;
    for (uint i = gid; i < p.N; i += tpg) {
        float x = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
        float y = float(target[i]);
        float l = -y * log(x) - (1.f - y) * log(1.f - x);
        acc += p.has_weight ? l * float(weight[i]) : l;
    }
    acc = tg_sum(acc, smem, lid, tgsz);
    if (lid == 0) partial[tgid] = acc * p.scale;
}

template <typename T>
kernel void bce_loss_bwd(
    device const T*  grad_out [[buffer(0)]],
    device const T*  input    [[buffer(1)]],
    device const T*  target   [[buffer(2)]],
    device const T*  weight   [[buffer(3)]],
    device       T*  grad_in  [[buffer(4)]],
    constant BCEParams& p     [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
    for (uint i = gid; i < p.N; i += tpg) {
        float x  = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
        float y  = float(target[i]);
        float dx = (x - y) / (x * (1.f - x));
        if (p.has_weight) dx *= float(weight[i]);
        float g = (p.reduction == 0) ? float(grad_out[i]) * dx : g_scalar * dx;
        grad_in[i] = T(g);
    }
}

#define INSTANTIATE_BCE(T)                                                    \
  template [[host_name("bce_loss_fwd_none_" #T)]]                           \
  kernel void bce_loss_fwd_none<T>(                                         \
      device const T*, device const T*, device const T*, device T*,        \
      constant BCEParams&, uint, uint);                                      \
  template [[host_name("bce_loss_fwd_reduce_" #T)]]                         \
  kernel void bce_loss_fwd_reduce<T>(                                       \
      device const T*, device const T*, device const T*, device float*,    \
      constant BCEParams&, uint, uint, uint, uint, uint);                   \
  template [[host_name("bce_loss_bwd_" #T)]]                                \
  kernel void bce_loss_bwd<T>(                                              \
      device const T*, device const T*, device const T*, device const T*,  \
      device T*, constant BCEParams&, uint, uint);

INSTANTIATE_BCE(float)
INSTANTIATE_BCE(half)
INSTANTIATE_BCE(bfloat)

// ============================================================================
// NLL Loss 1-D  (input (N,C) log-probs, target (N,) int class indices)
// C++ handles final scale (Mean denominator) after phase-2 reduction.
// Caller must pre-zero grad_in before dispatching nll_loss_bwd.
// ============================================================================

template <typename T>
kernel void nll_loss_fwd_none(
    device const T*     log_prob [[buffer(0)]],  // (N, C)
    device const int*   target   [[buffer(1)]],  // (N,)
    device const T*     weight   [[buffer(2)]],
    device       T*     out      [[buffer(3)]],  // (N,)
    constant NLLParams& p        [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    for (uint n = gid; n < p.N; n += tpg) {
        int t = target[n];
        if (t == p.ignore_index) { out[n] = T(0); continue; }
        float l = -float(log_prob[n * p.C + uint32_t(t)]);
        out[n] = T(p.has_weight ? l * float(weight[t]) : l);
    }
}

template <typename T>
kernel void nll_loss_fwd_reduce(
    device const T*     log_prob  [[buffer(0)]],
    device const int*   target    [[buffer(1)]],
    device const T*     weight    [[buffer(2)]],
    device       float* partial   [[buffer(3)]],  // (n_tg,) loss sums
    device       float* wpartial  [[buffer(4)]],  // (n_tg,) weight sums
    constant NLLParams& p         [[buffer(5)]],
    uint gid   [[thread_position_in_grid]],
    uint tpg   [[threads_per_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tgsz  [[threads_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]])
{
    threadgroup float smem[256], wsmem[256];
    float acc = 0.f, wacc = 0.f;
    for (uint n = gid; n < p.N; n += tpg) {
        int t = target[n];
        if (t == p.ignore_index) continue;
        float w = p.has_weight ? float(weight[t]) : 1.f;
        acc  += -float(log_prob[n * p.C + uint32_t(t)]) * w;
        wacc += w;
    }
    acc  = tg_sum(acc,  smem,  lid, tgsz);
    wacc = tg_sum(wacc, wsmem, lid, tgsz);
    if (lid == 0) {
        partial[tgid]  = acc;
        wpartial[tgid] = wacc;
    }
}

// Backward: writes -grad_out_scaled to grad_in[n, target[n]].
// Caller zeros grad_in before dispatch; each thread handles one n.
template <typename T>
kernel void nll_loss_bwd(
    device const T*     grad_out [[buffer(0)]],  // scalar (reduce) or (N,)
    device const int*   target   [[buffer(1)]],
    device const T*     weight   [[buffer(2)]],
    device       T*     grad_in  [[buffer(3)]],  // (N, C) pre-zeroed
    device const float* total_w  [[buffer(4)]],  // scalar weight sum (Mean)
    constant NLLParams& p        [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    for (uint n = gid; n < p.N; n += tpg) {
        int t = target[n];
        if (t == p.ignore_index) continue;
        float w = p.has_weight ? float(weight[t]) : 1.f;
        float scale;
        if (p.reduction == 0) {
            scale = -float(grad_out[n]) * w;
        } else if (p.reduction == 1) {
            float denom = (p.has_weight && total_w[0] > 0.f) ? total_w[0] : float(p.N);
            scale = -float(grad_out[0]) * w / denom;
        } else {
            scale = -float(grad_out[0]) * w;
        }
        grad_in[n * p.C + uint32_t(t)] = T(scale);
    }
}

#define INSTANTIATE_NLL(T)                                                    \
  template [[host_name("nll_loss_fwd_none_" #T)]]                           \
  kernel void nll_loss_fwd_none<T>(                                         \
      device const T*, device const int*, device const T*, device T*,      \
      constant NLLParams&, uint, uint);                                      \
  template [[host_name("nll_loss_fwd_reduce_" #T)]]                         \
  kernel void nll_loss_fwd_reduce<T>(                                       \
      device const T*, device const int*, device const T*,                  \
      device float*, device float*, constant NLLParams&,                    \
      uint, uint, uint, uint, uint);                                         \
  template [[host_name("nll_loss_bwd_" #T)]]                                \
  kernel void nll_loss_bwd<T>(                                              \
      device const T*, device const int*, device const T*, device T*,      \
      device const float*, constant NLLParams&, uint, uint);

INSTANTIATE_NLL(float)
INSTANTIATE_NLL(half)
INSTANTIATE_NLL(bfloat)

// ============================================================================
// Fused Cross-Entropy  (online Milakov-Gimelshein, one threadgroup per sample)
// lse_buf (N,) saved forward for backward; C++ reduces partial[] to scalar.
// ============================================================================

template <typename T>
kernel void cross_entropy_fwd(
    device const T*     logits   [[buffer(0)]],  // (N, C)
    device const int*   target   [[buffer(1)]],  // (N,)
    device       float* partial  [[buffer(2)]],  // (N,) per-sample loss
    device       float* lse_buf  [[buffer(3)]],  // (N,) log-sum-exp
    constant NLLParams& p        [[buffer(4)]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tgsz  [[threads_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]])
{
    uint n = tgid;
    if (n >= p.N) return;
    device const T* row = logits + n * p.C;

    threadgroup float smem_m[256];
    threadgroup float smem_s[256];

    // Per-thread online LSE (single pass, numerically stable)
    float lm = -INFINITY, ls = 0.f;
    for (uint c = lid; c < p.C; c += tgsz) {
        float x  = float(row[c]);
        float nm = max(lm, x);
        ls = ls * exp(lm - nm) + exp(x - nm);
        lm = nm;
    }

    // Tree-reduce (m, s) pairs
    smem_m[lid] = lm;
    smem_s[lid] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            float ma = smem_m[lid], mb = smem_m[lid + s];
            float sa = smem_s[lid], sb = smem_s[lid + s];
            float nm = max(ma, mb);
            smem_m[lid] = nm;
            smem_s[lid] = sa * exp(ma - nm) + sb * exp(mb - nm);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float lse = log(smem_s[0]) + smem_m[0];

    if (lid == 0) {
        int t      = target[n];
        float tv   = (t != p.ignore_index) ? float(row[t]) : 0.f;
        partial[n] = (t != p.ignore_index) ? lse - tv : 0.f;
        lse_buf[n] = lse;
    }
}

template <typename T>
kernel void cross_entropy_bwd(
    device const T*     grad_out  [[buffer(0)]],  // scalar or (N,)
    device const T*     logits    [[buffer(1)]],  // (N, C)
    device const int*   target    [[buffer(2)]],  // (N,)
    device const float* lse_buf   [[buffer(3)]],  // (N,)
    device       T*     grad_in   [[buffer(4)]],  // (N, C)
    constant NLLParams& p         [[buffer(5)]],
    constant float&     inv_scale [[buffer(6)]],  // 1/N (Mean) or 1.0 (Sum/None)
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]])
{
    for (uint idx = gid; idx < p.N * p.C; idx += tpg) {
        uint n   = idx / p.C;
        uint c   = idx % p.C;
        int  t   = target[n];
        float sm = exp(float(logits[idx]) - lse_buf[n]);
        float ind = (t != p.ignore_index && c == uint32_t(t)) ? 1.f : 0.f;
        float gout = (p.reduction == 0) ? float(grad_out[n]) : float(grad_out[0]);
        grad_in[idx] = T(gout * inv_scale * (sm - ind));
    }
}

#define INSTANTIATE_CE(T)                                                      \
  template [[host_name("cross_entropy_fwd_" #T)]]                             \
  kernel void cross_entropy_fwd<T>(                                           \
      device const T*, device const int*, device float*, device float*,       \
      constant NLLParams&, uint, uint, uint);                                  \
  template [[host_name("cross_entropy_bwd_" #T)]]                             \
  kernel void cross_entropy_bwd<T>(                                           \
      device const T*, device const T*, device const int*,                    \
      device const float*, device T*, constant NLLParams&,                    \
      constant float&, uint, uint);

INSTANTIATE_CE(float)
INSTANTIATE_CE(half)
INSTANTIATE_CE(bfloat)
