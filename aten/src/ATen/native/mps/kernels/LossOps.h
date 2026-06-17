#pragma once
#include <c10/metal/common.h>

// Shared by LossOps.metal (Metal kernels) and LossOps.mm (dispatch).
// The binary layout must stay identical on both sides.
struct BCEParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  uint32_t has_weight;
};
