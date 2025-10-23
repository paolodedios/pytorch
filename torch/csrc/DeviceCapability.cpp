#include <torch/csrc/DeviceCapability.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/core/DeviceCapability.h>
#include <c10/core/ScalarType.h>
#include <structmember.h>

PyTypeObject* THPDeviceCapabilityClass = nullptr;

static PyObject* THPDeviceCapability_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  
  // DeviceCapability is typically created from C++ side, not from Python
  // This constructor is mainly for internal use
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  
  THPDeviceCapability* self = (THPDeviceCapability*)ptr.get();
  // Initialize with default capabilities
  self->capability = c10::DeviceCapability();
  
  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDeviceCapability_Wrap(const c10::DeviceCapability& capability) {
  HANDLE_TH_ERRORS
  auto type = THPDeviceCapabilityClass;
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    throw python_error();
  }
  
  THPDeviceCapability* self = (THPDeviceCapability*)ptr.get();
  self->capability = capability;
  return ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPDeviceCapability_dealloc(THPDeviceCapability* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// Property getters for device capabilities
static PyObject* THPDeviceCapability_get_fp16(THPDeviceCapability* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->capability.has_fp16);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_get_fp32(THPDeviceCapability* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->capability.has_fp32);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_get_bf16(THPDeviceCapability* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->capability.has_bf16);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_get_int4(THPDeviceCapability* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->capability.has_int4);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_get_int8(THPDeviceCapability* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->capability.has_int8);
  END_HANDLE_TH_ERRORS
}

// Method implementations

static PyObject* THPDeviceCapability_repr(THPDeviceCapability* self) {
  HANDLE_TH_ERRORS
  std::string repr = "DeviceCapability(";
  repr += "fp16=" + std::to_string(self->capability.has_fp16) + ", ";
  repr += "fp32=" + std::to_string(self->capability.has_fp32) + ", ";
  repr += "bf16=" + std::to_string(self->capability.has_bf16) + ", ";
  repr += "int4=" + std::to_string(self->capability.has_int4) + ", ";
  repr += "int8=" + std::to_string(self->capability.has_int8);
  repr += ")";
  return THPUtils_packString(repr);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_eq(PyObject* self, PyObject* other) {
  HANDLE_TH_ERRORS
  if (!THPDeviceCapability_Check(self) || !THPDeviceCapability_Check(other)) {
    return PyBool_FromLong(0);
  }
  THPDeviceCapability* self_cap = (THPDeviceCapability*)self;
  THPDeviceCapability* other_cap = (THPDeviceCapability*)other;
  return PyBool_FromLong(
      self_cap->capability.capability_bits == other_cap->capability.capability_bits);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_ne(PyObject* self, PyObject* other) {
  HANDLE_TH_ERRORS
  if (!THPDeviceCapability_Check(self) || !THPDeviceCapability_Check(other)) {
    return PyBool_FromLong(1);
  }
  THPDeviceCapability* self_cap = (THPDeviceCapability*)self;
  THPDeviceCapability* other_cap = (THPDeviceCapability*)other;
  return PyBool_FromLong(
      self_cap->capability.capability_bits != other_cap->capability.capability_bits);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDeviceCapability_richcompare(
    PyObject* self,
    PyObject* other,
    int op) {
  PyObject* result = nullptr;
  if (other == Py_None) {
    result = Py_False;
  } else {
    switch (op) {
      case Py_EQ:
        result = THPDeviceCapability_eq(self, other);
        break;
      case Py_NE:
        result = THPDeviceCapability_ne(self, other);
        break;
      default:
        result = Py_False;
        break;
    }
  }
  Py_XINCREF(result);
  return result;
}

static const std::initializer_list<PyGetSetDef> THPDeviceCapability_properties = {
    {"has_fp16", (getter)THPDeviceCapability_get_fp16, nullptr, nullptr, nullptr},
    {"has_fp32", (getter)THPDeviceCapability_get_fp32, nullptr, nullptr, nullptr},
    {"has_bf16", (getter)THPDeviceCapability_get_bf16, nullptr, nullptr, nullptr},
    {"has_int4", (getter)THPDeviceCapability_get_int4, nullptr, nullptr, nullptr},
    {"has_int8", (getter)THPDeviceCapability_get_int8, nullptr, nullptr, nullptr},
    {nullptr}};

static const std::initializer_list<PyMethodDef> THPDeviceCapability_methods = {
    {"__eq__", THPDeviceCapability_eq, METH_O, nullptr},
    {nullptr}};

static PyTypeObject THPDeviceCapabilityType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.DeviceCapability", /* tp_name */
    sizeof(THPDeviceCapability), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPDeviceCapability_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPDeviceCapability_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    THPDeviceCapability_richcompare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    const_cast<PyMethodDef*>(std::data(THPDeviceCapability_methods)), /* tp_methods */
    nullptr, /* tp_members */
    const_cast<PyGetSetDef*>(std::data(THPDeviceCapability_properties)), /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPDeviceCapability_pynew, /* tp_new */
};

void THPDeviceCapability_init(PyObject* module) {
  THPDeviceCapabilityClass = &THPDeviceCapabilityType;
  Py_SET_TYPE(&THPDeviceCapabilityType, &PyType_Type);
  if (PyType_Ready(&THPDeviceCapabilityType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDeviceCapabilityType);
  if (PyModule_AddObject(module, "DeviceCapability", (PyObject*)&THPDeviceCapabilityType) < 0) {
    throw python_error();
  }
}
