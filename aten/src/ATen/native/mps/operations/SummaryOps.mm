//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bincount_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Bincount_metallib.h>
#endif

static std::string idx_dtype_name(ScalarType st) {
  switch (st) {
    case ScalarType::Char:
      return "char";
    case ScalarType::Short:
      return "short";
    case ScalarType::Int:
      return "int";
    case ScalarType::Long:
      return "long";
    case ScalarType::Byte:
      return "uchar";
    default:
      TORCH_CHECK(false, "bincount: unsupported index dtype ", st);
  }
}

static Tensor bincount_mps_unweighted(const Tensor& self, int64_t nbins) {
  // Two-stage approach: first accumulate into a uint32 atomic scratch buffer,
  // then widen into the int64 output that the bincount op promises.
  Tensor counts_u32 = at::zeros({nbins}, self.options().dtype(kInt));
  Tensor output = at::empty({nbins}, self.options().dtype(kLong));

  const uint64_t numel = static_cast<uint64_t>(self.numel());
  const uint64_t nbins_u = static_cast<uint64_t>(nbins);

  const std::string add_key = "bincount_unweighted_" + idx_dtype_name(self.scalar_type());
  const std::string widen_key = "bincount_widen_uint_to_long";

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> add_pso = lib.getPipelineStateForFunc(add_key);
      id<MTLComputePipelineState> widen_pso = lib.getPipelineStateForFunc(widen_key);

      getMPSProfiler().beginProfileKernel(add_pso, add_key, false);
      [encoder setComputePipelineState:add_pso];
      mtl_setArgs(encoder, self, counts_u32, numel);
      mtl_dispatch1DJob(encoder, add_pso, static_cast<NSUInteger>(numel));
      getMPSProfiler().endProfileKernel(add_pso);

      getMPSProfiler().beginProfileKernel(widen_pso, widen_key, false);
      [encoder setComputePipelineState:widen_pso];
      mtl_setArgs(encoder, counts_u32, output, nbins_u);
      mtl_dispatch1DJob(encoder, widen_pso, static_cast<NSUInteger>(nbins_u));
      getMPSProfiler().endProfileKernel(widen_pso);
    }
  });

  return output;
}

static Tensor bincount_mps_weighted(const Tensor& self, const Tensor& weights, int64_t nbins) {
  // Output dtype matches the prior MPSGraph implementation's contract:
  //   Float           -> Float       accumulator: atomic<float>
  //   Int             -> Int         accumulator: atomic<int>
  //   Half / BFloat16 -> input dtype accumulator: atomic<float> + final cast
  //   everything else -> Int         accumulator: atomic<int>  (matches the
  //                                  legacy MPSGraph path which truncated
  //                                  fractional weights — see test_bincount_reduction).
  const ScalarType weight_dtype = weights.scalar_type();

  std::string kernel_kind;
  Tensor weights_for_kernel = weights;
  ScalarType output_dtype;
  ScalarType accum_dtype;
  bool cast_back_to_input_dtype = false;

  if (weight_dtype == kFloat) {
    kernel_kind = "bincount_weighted_float_";
    accum_dtype = kFloat;
    output_dtype = kFloat;
  } else if (weight_dtype == kInt) {
    kernel_kind = "bincount_weighted_int_";
    accum_dtype = kInt;
    output_dtype = kInt;
  } else if (weight_dtype == kHalf || weight_dtype == kBFloat16) {
    kernel_kind = "bincount_weighted_float_";
    weights_for_kernel = weights.to(kFloat);
    accum_dtype = kFloat;
    output_dtype = weight_dtype;
    cast_back_to_input_dtype = true;
  } else {
    kernel_kind = "bincount_weighted_int_";
    weights_for_kernel = weights.to(kInt);
    accum_dtype = kInt;
    output_dtype = kInt;
  }

  Tensor accum = at::zeros({nbins}, weights.options().dtype(accum_dtype));
  const std::string key = kernel_kind + idx_dtype_name(self.scalar_type());
  const uint64_t numel = static_cast<uint64_t>(self.numel());

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);
      getMPSProfiler().beginProfileKernel(pso, key, false);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self, weights_for_kernel, accum, numel);
      mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(numel));
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  if (cast_back_to_input_dtype) {
    return accum.to(output_dtype);
  }
  return accum;
}

Tensor _bincount_mps(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  TORCH_CHECK(c10::isIntegralType(self.scalar_type(), /*includesBool=*/true));
  TORCH_CHECK(minlength >= 0, "minlength should be >= 0");

  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong, std::nullopt, kMPS, std::nullopt);
  }
  TORCH_CHECK(self.dim() == 1, "bincount only supports 1-d non-negative integral inputs.");
  TORCH_CHECK(self.scalar_type() != kBool, "bincount is not supported for Bool");

  bool has_weights = weights.defined();
  TORCH_CHECK(!(has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))),
              "weights should be 1-d and have the same length as input");
  TORCH_CHECK(self.numel() <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
              "bincount on MPS supports inputs with at most 2^32-1 elements");

  // Single sync to derive the bin count. Both reductions go through MPS, but
  // we batch them so it's at most two short reductions of the input tensor.
  Tensor self_contig = self.contiguous();
  const int64_t input_max = self_contig.max().item<int64_t>();
  const int64_t input_min = self_contig.min().item<int64_t>();
  TORCH_CHECK(input_min >= 0, "bincount only supports 1-d non-negative integral inputs.");

  const int64_t nbins = std::max(input_max + 1, minlength);

  Tensor input_for_kernel = self_contig;
  if (self_contig.scalar_type() == kBool) {
    input_for_kernel = self_contig.to(kInt);
  }

  if (has_weights) {
    return bincount_mps_weighted(input_for_kernel, weights.contiguous(), nbins);
  } else {
    return bincount_mps_unweighted(input_for_kernel, nbins);
  }
}

} // namespace at::native
