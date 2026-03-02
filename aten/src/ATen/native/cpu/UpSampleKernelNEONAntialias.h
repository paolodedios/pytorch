#pragma once
#if defined(__aarch64__)

#include <ATen/core/Tensor.h>
#include <arm_neon.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace {

// Emulate SSE _mm_madd_epi16: multiply int16 pairs and add adjacent results
inline int32x4_t neon_madd_s16(int16x8_t a, int16x8_t b) {
  int32x4_t prod_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
  int32x4_t prod_high = vmull_s16(vget_high_s16(a), vget_high_s16(b));
  return vpaddq_s32(prod_low, prod_high);
}

// Forward declarations
void NeonResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels);

void NeonResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels);

void NeonResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels);

void NeonResampleHorizontal(
    const at::Tensor& unpacked_output,
    const at::Tensor& unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& horiz_indices_weights,
    unsigned int horiz_weights_precision) {

  const auto* kk =
      (const int16_t*)(horiz_indices_weights[3].const_data_ptr<double>());

  auto xout = unpacked_output.size(2);
  auto yin = unpacked_output.size(1);
  auto xin = unpacked_input.size(2);
  const auto num_channels = unpacked_input.size(0);
  TORCH_INTERNAL_ASSERT(num_channels == 3);

  const int64_t* idx_ptr_xmin =
      horiz_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size =
      horiz_indices_weights[1].const_data_ptr<int64_t>();

  uint8_t* output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* input_p = unpacked_input.const_data_ptr<uint8_t>();

  auto xout_stride = xout * num_channels;
  auto xin_stride = xin * num_channels;

  int64_t yy = 0;
  for (; yy < yin - 3; yy += 4) {
    NeonResampleHorizontalConvolution8u4x(
        output_p + yy * xout_stride,
        output_p + (yy + 1) * xout_stride,
        output_p + (yy + 2) * xout_stride,
        output_p + (yy + 3) * xout_stride,
        xout,
        input_p + yy * xin_stride,
        input_p + (yy + 1) * xin_stride,
        input_p + (yy + 2) * xin_stride,
        input_p + (yy + 3) * xin_stride,
        xin,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels);
  }
  for (; yy < yin; yy++) {
    NeonResampleHorizontalConvolution8u(
        output_p + yy * xout_stride,
        xout,
        input_p + yy * xin_stride,
        xin,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels);
  }
}

void NeonResampleVertical(
    const at::Tensor& unpacked_output,
    const at::Tensor& unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& vert_indices_weights,
    unsigned int vert_weights_precision) {

  const auto* kk =
      (const int16_t*)(vert_indices_weights[3].const_data_ptr<double>());

  const int64_t* idx_ptr_xmin =
      vert_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size =
      vert_indices_weights[1].const_data_ptr<int64_t>();

  uint8_t* output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* input_p = unpacked_input.const_data_ptr<uint8_t>();

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  const auto num_channels = unpacked_input.size(0);
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_output.size(0));

  auto xout_stride = xout * num_channels;
  for (const auto yy : c10::irange(yout)) {
    const auto* k = &kk[yy * ksize];
    auto ids_min = idx_ptr_xmin[yy];
    auto ids_size = idx_ptr_size[yy];
    NeonResampleVerticalConvolution8u(
        output_p + yy * xout_stride,
        input_p,
        xout,
        ids_min,
        ids_size,
        k,
        vert_weights_precision,
        num_channels);
  }
}

// Main entry point. Supports bilinear or bicubic mode for uint8 dtype with
// num_channels == 3 and channels-last memory format, with or without antialias.
// Mirrors upsample_avx_bilinear_bicubic_uint8 but uses NEON intrinsics and
// works directly on interleaved RGB data (no unpack/pack needed).
template <typename scale_type, class F>
void upsample_neon_bilinear_bicubic_uint8(
    const at::Tensor& input_,
    const at::Tensor& output,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

  auto batch_size = input_.size(0);
  auto num_channels = input_.size(1);
  auto xin = input_.size(3);
  auto yin = input_.size(2);
  auto xout = output.size(3);
  auto yout = output.size(2);

  if (xin == xout && yin == yout) {
    output.copy_(input_);
    return;
  }

  TORCH_INTERNAL_ASSERT(num_channels == 3);
  TORCH_INTERNAL_ASSERT(output.is_contiguous(at::MemoryFormat::ChannelsLast));

  auto input = input_.contiguous(at::MemoryFormat::ChannelsLast);

  auto need_horizontal = xout != xin;
  auto need_vertical = yout != yin;

  int ksize_horiz, ksize_vert;
  std::vector<at::Tensor> horiz_indices_weights, vert_indices_weights;
  unsigned int horiz_weights_precision, vert_weights_precision;

  if (need_horizontal) {
    int interp_dim = 3;
    std::tie(horiz_indices_weights, ksize_horiz, horiz_weights_precision) =
        F::compute_index_ranges_int16_weights(
            /*input_size=*/xin,
            /*output_size=*/xout,
            /*stride=*/num_channels,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/false);
  }

  if (need_vertical) {
    int interp_dim = 2;
    std::tie(vert_indices_weights, ksize_vert, vert_weights_precision) =
        F::compute_index_ranges_int16_weights(
            /*input_size=*/yin,
            /*output_size=*/yout,
            /*stride=*/num_channels * xout,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/false);
  }

  at::Tensor buffer_horiz;
  if (need_horizontal && need_vertical) {
    buffer_horiz = at::empty({num_channels, yin, xout}, input.options());
  }

  for (const auto i : c10::irange(batch_size)) {
    at::Tensor input_slice = input[i];

    if (need_horizontal) {
      at::Tensor horiz_output = need_vertical ? buffer_horiz : output[i];
      NeonResampleHorizontal(
          horiz_output,
          input_slice,
          ksize_horiz,
          horiz_indices_weights,
          horiz_weights_precision);
      if (need_vertical) {
        input_slice = horiz_output;
      }
    }
    if (need_vertical) {
      NeonResampleVertical(
          output[i],
          input_slice,
          ksize_vert,
          vert_indices_weights,
          vert_weights_precision);
    }
  }
}

// ---- Convolution implementations ----

void NeonResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const int64_t stride = num_channels;
  const int64_t data_size = xsize * stride;
  const int32_t initial_val = 1 << (coefs_precision - 1);
  const int32x4_t initial = vdupq_n_s32(initial_val);

  int64_t j = 0;

  // Process 16 bytes at a time
  for (; j + 16 <= data_size; j += 16) {
    int32x4_t sss0 = initial;
    int32x4_t sss1 = initial;
    int32x4_t sss2 = initial;
    int32x4_t sss3 = initial;

    const uint8_t* lineIn_min = lineIn + j + ids_min;

    // Process 2 weights at a time
    int64_t i = 0;
    for (; i + 1 < ids_size; i += 2) {
      int16_t w0 = k[i];
      int16_t w1 = k[i + 1];
      int16x8_t mmk = {w0, w1, w0, w1, w0, w1, w0, w1};

      uint8x16_t src1 = vld1q_u8(lineIn_min + i * data_size);
      uint8x16_t src2 = vld1q_u8(lineIn_min + (i + 1) * data_size);

      uint8x16x2_t interleaved = vzipq_u8(src1, src2);

      int16x8_t pix0 = vreinterpretq_s16_u16(
          vmovl_u8(vget_low_u8(interleaved.val[0])));
      int16x8_t pix1 = vreinterpretq_s16_u16(
          vmovl_u8(vget_high_u8(interleaved.val[0])));
      int16x8_t pix2 = vreinterpretq_s16_u16(
          vmovl_u8(vget_low_u8(interleaved.val[1])));
      int16x8_t pix3 = vreinterpretq_s16_u16(
          vmovl_u8(vget_high_u8(interleaved.val[1])));

      sss0 = vaddq_s32(sss0, neon_madd_s16(pix0, mmk));
      sss1 = vaddq_s32(sss1, neon_madd_s16(pix1, mmk));
      sss2 = vaddq_s32(sss2, neon_madd_s16(pix2, mmk));
      sss3 = vaddq_s32(sss3, neon_madd_s16(pix3, mmk));
    }

    // Handle remaining single weight
    for (; i < ids_size; i++) {
      int16x8_t mmk = vdupq_n_s16(k[i]);

      uint8x16_t src = vld1q_u8(lineIn_min + i * data_size);

      int16x8_t pix_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src)));
      int16x8_t pix_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src)));

      sss0 = vaddq_s32(
          sss0, vmull_s16(vget_low_s16(pix_lo), vget_low_s16(mmk)));
      sss1 = vaddq_s32(
          sss1, vmull_s16(vget_high_s16(pix_lo), vget_high_s16(mmk)));
      sss2 = vaddq_s32(
          sss2, vmull_s16(vget_low_s16(pix_hi), vget_low_s16(mmk)));
      sss3 = vaddq_s32(
          sss3, vmull_s16(vget_high_s16(pix_hi), vget_high_s16(mmk)));
    }

    int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(coefs_precision));
    sss0 = vshlq_s32(sss0, shift);
    sss1 = vshlq_s32(sss1, shift);
    sss2 = vshlq_s32(sss2, shift);
    sss3 = vshlq_s32(sss3, shift);

    int16x8_t narrow_lo = vcombine_s16(vqmovn_s32(sss0), vqmovn_s32(sss1));
    int16x8_t narrow_hi = vcombine_s16(vqmovn_s32(sss2), vqmovn_s32(sss3));

    vst1_u8(lineOut + j, vqmovun_s16(narrow_lo));
    vst1_u8(lineOut + j + 8, vqmovun_s16(narrow_hi));
  }

  // Scalar fallback for remaining bytes
  for (; j < data_size; j++) {
    int32_t sss = initial_val;
    const uint8_t* lineIn_min = lineIn + j + ids_min;

    for (int64_t i = 0; i < ids_size; i++) {
      sss += k[i] * static_cast<int32_t>(lineIn_min[i * data_size]);
    }

    sss >>= coefs_precision;
    lineOut[j] =
        static_cast<uint8_t>(std::clamp(sss, 0, 255));
  }
}

void NeonResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const int64_t stride = num_channels;
  const int32_t initial_val = 1 << (coefs_precision - 1);

  for (int64_t out_x = 0; out_x < out_xsize; out_x++) {
    const int64_t ids_min = idx_ptr_xmin[out_x];
    const int64_t ids_size = idx_ptr_size[out_x];
    const int16_t* k = &kk[out_x * kmax];

    const uint8_t* lineIn0_min = lineIn0 + ids_min;
    const uint8_t* lineIn1_min = lineIn1 + ids_min;
    const uint8_t* lineIn2_min = lineIn2 + ids_min;
    const uint8_t* lineIn3_min = lineIn3 + ids_min;

    int32x4_t acc0_r = vdupq_n_s32(0), acc0_g = vdupq_n_s32(0),
              acc0_b = vdupq_n_s32(0);
    int32x4_t acc1_r = vdupq_n_s32(0), acc1_g = vdupq_n_s32(0),
              acc1_b = vdupq_n_s32(0);
    int32x4_t acc2_r = vdupq_n_s32(0), acc2_g = vdupq_n_s32(0),
              acc2_b = vdupq_n_s32(0);
    int32x4_t acc3_r = vdupq_n_s32(0), acc3_g = vdupq_n_s32(0),
              acc3_b = vdupq_n_s32(0);

    int64_t i = 0;

    // Process 8 pixels at a time: load weights once, apply to 4 rows
    for (; i + 8 <= ids_size; i += 8) {
      int16x8_t weights = vld1q_s16(&k[i]);
      int16x4_t w_lo = vget_low_s16(weights);
      int16x4_t w_hi = vget_high_s16(weights);

      // Row 0
      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      int16x8_t r0 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[0]));
      int16x8_t g0 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[1]));
      int16x8_t b0 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[2]));
      acc0_r = vmlal_s16(acc0_r, vget_low_s16(r0), w_lo);
      acc0_r = vmlal_s16(acc0_r, vget_high_s16(r0), w_hi);
      acc0_g = vmlal_s16(acc0_g, vget_low_s16(g0), w_lo);
      acc0_g = vmlal_s16(acc0_g, vget_high_s16(g0), w_hi);
      acc0_b = vmlal_s16(acc0_b, vget_low_s16(b0), w_lo);
      acc0_b = vmlal_s16(acc0_b, vget_high_s16(b0), w_hi);

      // Row 1
      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      int16x8_t r1 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[0]));
      int16x8_t g1 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[1]));
      int16x8_t b1 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[2]));
      acc1_r = vmlal_s16(acc1_r, vget_low_s16(r1), w_lo);
      acc1_r = vmlal_s16(acc1_r, vget_high_s16(r1), w_hi);
      acc1_g = vmlal_s16(acc1_g, vget_low_s16(g1), w_lo);
      acc1_g = vmlal_s16(acc1_g, vget_high_s16(g1), w_hi);
      acc1_b = vmlal_s16(acc1_b, vget_low_s16(b1), w_lo);
      acc1_b = vmlal_s16(acc1_b, vget_high_s16(b1), w_hi);

      // Row 2
      uint8x8x3_t rgb2 = vld3_u8(lineIn2_min + stride * i);
      int16x8_t r2 = vreinterpretq_s16_u16(vmovl_u8(rgb2.val[0]));
      int16x8_t g2 = vreinterpretq_s16_u16(vmovl_u8(rgb2.val[1]));
      int16x8_t b2 = vreinterpretq_s16_u16(vmovl_u8(rgb2.val[2]));
      acc2_r = vmlal_s16(acc2_r, vget_low_s16(r2), w_lo);
      acc2_r = vmlal_s16(acc2_r, vget_high_s16(r2), w_hi);
      acc2_g = vmlal_s16(acc2_g, vget_low_s16(g2), w_lo);
      acc2_g = vmlal_s16(acc2_g, vget_high_s16(g2), w_hi);
      acc2_b = vmlal_s16(acc2_b, vget_low_s16(b2), w_lo);
      acc2_b = vmlal_s16(acc2_b, vget_high_s16(b2), w_hi);

      // Row 3
      uint8x8x3_t rgb3 = vld3_u8(lineIn3_min + stride * i);
      int16x8_t r3 = vreinterpretq_s16_u16(vmovl_u8(rgb3.val[0]));
      int16x8_t g3 = vreinterpretq_s16_u16(vmovl_u8(rgb3.val[1]));
      int16x8_t b3 = vreinterpretq_s16_u16(vmovl_u8(rgb3.val[2]));
      acc3_r = vmlal_s16(acc3_r, vget_low_s16(r3), w_lo);
      acc3_r = vmlal_s16(acc3_r, vget_high_s16(r3), w_hi);
      acc3_g = vmlal_s16(acc3_g, vget_low_s16(g3), w_lo);
      acc3_g = vmlal_s16(acc3_g, vget_high_s16(g3), w_hi);
      acc3_b = vmlal_s16(acc3_b, vget_low_s16(b3), w_lo);
      acc3_b = vmlal_s16(acc3_b, vget_high_s16(b3), w_hi);
    }

    // Process 4 pixels at a time
    for (; i + 4 <= ids_size; i += 4) {
      int16x4_t w4 = vld1_s16(&k[i]);

      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      acc0_r = vmlal_s16(acc0_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[0]))), w4);
      acc0_g = vmlal_s16(acc0_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[1]))), w4);
      acc0_b = vmlal_s16(acc0_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[2]))), w4);

      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      acc1_r = vmlal_s16(acc1_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[0]))), w4);
      acc1_g = vmlal_s16(acc1_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[1]))), w4);
      acc1_b = vmlal_s16(acc1_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[2]))), w4);

      uint8x8x3_t rgb2 = vld3_u8(lineIn2_min + stride * i);
      acc2_r = vmlal_s16(acc2_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb2.val[0]))), w4);
      acc2_g = vmlal_s16(acc2_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb2.val[1]))), w4);
      acc2_b = vmlal_s16(acc2_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb2.val[2]))), w4);

      uint8x8x3_t rgb3 = vld3_u8(lineIn3_min + stride * i);
      acc3_r = vmlal_s16(acc3_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb3.val[0]))), w4);
      acc3_g = vmlal_s16(acc3_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb3.val[1]))), w4);
      acc3_b = vmlal_s16(acc3_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb3.val[2]))), w4);
    }

    int32_t s0_r = vaddvq_s32(acc0_r) + initial_val;
    int32_t s0_g = vaddvq_s32(acc0_g) + initial_val;
    int32_t s0_b = vaddvq_s32(acc0_b) + initial_val;
    int32_t s1_r = vaddvq_s32(acc1_r) + initial_val;
    int32_t s1_g = vaddvq_s32(acc1_g) + initial_val;
    int32_t s1_b = vaddvq_s32(acc1_b) + initial_val;
    int32_t s2_r = vaddvq_s32(acc2_r) + initial_val;
    int32_t s2_g = vaddvq_s32(acc2_g) + initial_val;
    int32_t s2_b = vaddvq_s32(acc2_b) + initial_val;
    int32_t s3_r = vaddvq_s32(acc3_r) + initial_val;
    int32_t s3_g = vaddvq_s32(acc3_g) + initial_val;
    int32_t s3_b = vaddvq_s32(acc3_b) + initial_val;

    // Scalar cleanup for remaining pixels
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p0 = lineIn0_min + stride * i;
      const uint8_t* p1 = lineIn1_min + stride * i;
      const uint8_t* p2 = lineIn2_min + stride * i;
      const uint8_t* p3 = lineIn3_min + stride * i;
      s0_r += w * p0[0]; s0_g += w * p0[1]; s0_b += w * p0[2];
      s1_r += w * p1[0]; s1_g += w * p1[1]; s1_b += w * p1[2];
      s2_r += w * p2[0]; s2_g += w * p2[1]; s2_b += w * p2[2];
      s3_r += w * p3[0]; s3_g += w * p3[1]; s3_b += w * p3[2];
    }

    auto clamp_and_shift = [coefs_precision](int32_t v) -> uint8_t {
      return static_cast<uint8_t>(std::clamp(v >> coefs_precision, 0, 255));
    };

    uint8_t* o0 = lineOut0 + stride * out_x;
    uint8_t* o1 = lineOut1 + stride * out_x;
    uint8_t* o2 = lineOut2 + stride * out_x;
    uint8_t* o3 = lineOut3 + stride * out_x;
    o0[0] = clamp_and_shift(s0_r);
    o0[1] = clamp_and_shift(s0_g);
    o0[2] = clamp_and_shift(s0_b);
    o1[0] = clamp_and_shift(s1_r);
    o1[1] = clamp_and_shift(s1_g);
    o1[2] = clamp_and_shift(s1_b);
    o2[0] = clamp_and_shift(s2_r);
    o2[1] = clamp_and_shift(s2_g);
    o2[2] = clamp_and_shift(s2_b);
    o3[0] = clamp_and_shift(s3_r);
    o3[1] = clamp_and_shift(s3_g);
    o3[2] = clamp_and_shift(s3_b);
  }
}

void NeonResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const int64_t stride = num_channels;
  const int32_t initial_val = 1 << (coefs_precision - 1);

  for (int64_t out_x = 0; out_x < out_xsize; out_x++) {
    const int64_t ids_min = idx_ptr_xmin[out_x];
    const int64_t ids_size = idx_ptr_size[out_x];
    const int16_t* k = &kk[out_x * kmax];
    const uint8_t* lineIn_min = lineIn + ids_min;

    int32x4_t acc_r = vdupq_n_s32(0);
    int32x4_t acc_g = vdupq_n_s32(0);
    int32x4_t acc_b = vdupq_n_s32(0);

    int64_t i = 0;

    // Process 8 pixels at a time using vld3 to deinterleave RGB
    for (; i + 8 <= ids_size; i += 8) {
      uint8x8x3_t rgb = vld3_u8(lineIn_min + stride * i);
      int16x8_t weights = vld1q_s16(&k[i]);

      int16x8_t r16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[0]));
      int16x8_t g16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[1]));
      int16x8_t b16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[2]));

      acc_r = vmlal_s16(acc_r, vget_low_s16(r16), vget_low_s16(weights));
      acc_r = vmlal_s16(acc_r, vget_high_s16(r16), vget_high_s16(weights));
      acc_g = vmlal_s16(acc_g, vget_low_s16(g16), vget_low_s16(weights));
      acc_g = vmlal_s16(acc_g, vget_high_s16(g16), vget_high_s16(weights));
      acc_b = vmlal_s16(acc_b, vget_low_s16(b16), vget_low_s16(weights));
      acc_b = vmlal_s16(acc_b, vget_high_s16(b16), vget_high_s16(weights));
    }

    // Process 4 pixels at a time
    for (; i + 4 <= ids_size; i += 4) {
      uint8x8x3_t rgb = vld3_u8(lineIn_min + stride * i);
      int16x4_t w4 = vld1_s16(&k[i]);

      acc_r = vmlal_s16(
          acc_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[0]))), w4);
      acc_g = vmlal_s16(
          acc_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[1]))), w4);
      acc_b = vmlal_s16(
          acc_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[2]))), w4);
    }

    int32_t sum_r = vaddvq_s32(acc_r) + initial_val;
    int32_t sum_g = vaddvq_s32(acc_g) + initial_val;
    int32_t sum_b = vaddvq_s32(acc_b) + initial_val;

    // Scalar cleanup for remaining pixels
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p = lineIn_min + stride * i;
      sum_r += w * p[0];
      sum_g += w * p[1];
      sum_b += w * p[2];
    }

    uint8_t* out = lineOut + stride * out_x;
    out[0] = static_cast<uint8_t>(std::clamp(sum_r >> coefs_precision, 0, 255));
    out[1] = static_cast<uint8_t>(std::clamp(sum_g >> coefs_precision, 0, 255));
    out[2] = static_cast<uint8_t>(std::clamp(sum_b >> coefs_precision, 0, 255));
  }
}

} // anonymous namespace

#endif // __aarch64__
