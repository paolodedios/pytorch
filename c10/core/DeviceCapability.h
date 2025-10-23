#pragma once

#include <c10/macros/Export.h>
#include <c10/core/ScalarType.h>

namespace c10 {

#define AT_FORALL_DEVICE_CAPABILITIES(_) \
  /* Data precision support */           \
  _(fp16)                                \
  _(fp32)                                \
  _(bf16)                                \
  _(int4)                                \
  _(int8)

#define NUMBER_OF_DEVICE_CAPABILITIES 5
#define ALL_CAPABILITIES_MASK() ((1 << NUMBER_OF_DEVICE_CAPABILITIES) - 1)

#define DEFINE_DEVICE_CAPABILITY(capability) unsigned int has_##capability : 1;

struct C10_API DeviceCapability {
  union {
    struct {
      AT_FORALL_DEVICE_CAPABILITIES(DEFINE_DEVICE_CAPABILITY)
    };
    unsigned int capability_bits; // Allow direct bit manipulation
  };

  // Default constructor with all capabilities enabled.
  DeviceCapability() : capability_bits(ALL_CAPABILITIES_MASK()) {}

};

#undef AT_FORALL_DEVICE_CAPABILITIES
#undef DEFINE_DEVICE_CAPABILITY
#undef ALL_CAPABILITIES_MASK
#undef NUMBER_OF_DEVICE_CAPABILITIES

} // namespace c10
