#pragma once
#include <c10/metal/common.h>

// Shared by LossOps.metal (Metal kernels) and LossOps.mm (dispatch).
// The binary layout must stay identical on both sides.
struct SmoothHuberParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  float beta;
  uint32_t is_huber; // 0=SmoothL1, 1=HuberLoss
};
