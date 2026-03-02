#pragma once

#include <c10/metal/common.h>

enum class PdistMode : int32_t {
  MODE_ZERO = 0,
  MODE_ONE = 1,
  MODE_TWO = 2,
  MODE_INF = 3,
  MODE_GENERAL = 4,
  MODE_LT_TWO = 5,
};

#ifndef __METAL__
#include <cmath>

inline PdistMode pdist_mode(double p, bool backward) {
  if (p == 1.0) {
    return PdistMode::MODE_ONE;
  }
  if (p == 2.0) {
    return PdistMode::MODE_TWO;
  }
  if (std::isinf(p)) {
    return PdistMode::MODE_INF;
  }

  if (!backward && p == 0.0) {
    return PdistMode::MODE_ZERO;
  }
  if (backward && p < 2.0) {
    return PdistMode::MODE_LT_TWO;
  }

  return PdistMode::MODE_GENERAL;
}
#endif
