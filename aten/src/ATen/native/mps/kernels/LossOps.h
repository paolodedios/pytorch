#pragma once
#include <c10/metal/common.h>

// Shared by LossOps.metal (Metal kernels) and LossOps.mm (dispatch).
// The binary layout must stay identical on both sides.
struct NLLParams {
  uint32_t N;
  uint32_t C;
  int32_t ignore_index;
  uint32_t reduction;
  uint32_t has_weight;
};
