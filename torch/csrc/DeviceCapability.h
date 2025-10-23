#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>
#include <c10/core/DeviceCapability.h>

struct THPDeviceCapability {
  PyObject_HEAD
  c10::DeviceCapability capability;
};

extern TORCH_API PyTypeObject* THPDeviceCapabilityClass;

void THPDeviceCapability_init(PyObject* module);

inline bool THPDeviceCapability_Check(PyObject* obj) {
  return THPDeviceCapabilityClass && PyObject_IsInstance(obj, (PyObject*)THPDeviceCapabilityClass);
}

TORCH_PYTHON_API PyObject* THPDeviceCapability_Wrap(const c10::DeviceCapability& capability);
