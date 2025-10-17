// Compile this file as C, not C++.

// Wrap inclusion of pycore_stackref.h inside a pure C file and ensure this file
// is compiled as C (not C++). This avoids MSVC’s “designated initializers
// require /std:c++20” error, since pycore_stackref.h uses C99-style designated
// initializers that are not supported in older C++ standards, but is supported
// in C.

#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_14_PLUS
#pragma message("is Python 3.14+")
#endif

#if defined(_WIN32)
#pragma message("WIN32 defined")
#endif

#if IS_PYTHON_3_14_PLUS && defined(_WIN32)

#define Py_BUILD_CORE
#include <Python.h>
#include <internal/pycore_stackref.h>
#include "stackref_bridge.h"
#undef Py_BUILD_CORE


PyObject* Torch_PyStackRef_AsPyObjectBorrow(void* stackref) {
  _PyStackRef *sr = (_PyStackRef*)stackref;
  return PyStackRef_AsPyObjectBorrow(*sr);
}

#endif
