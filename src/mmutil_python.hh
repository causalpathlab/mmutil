#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "io.hh"
#include "mmutil.hh"
#include "utils/util.hh"

#ifndef MMUTIL_PYTHON_HH_
#define MMUTIL_PYTHON_HH_

// Already declared in mmutil.hh:
// using Index = size_t;
// using Scalar = float;

using Triplet    = std::tuple<Index, Index, Scalar>;
using TripletVec = std::vector<Triplet>;

auto
make_argv(const PyObject* args) {
  const Index argc = PyTuple_GET_SIZE(args);
  std::vector<PyObject*> argv(argc);

  for (Index i = 0; i < argc; ++i) {
    argv[i] = PyTuple_GET_ITEM(args, i);
  }

  return argv;
}

inline std::string
pyobj_string(PyObject* obj) {
  return PyBytes_AsString(PyUnicode_AsEncodedString(obj, "UTF-8", "strict"));
}

std::vector<std::string>
pyobj_string_vector(PyObject* listObj) {
  const int _sz = PyList_Size(listObj);
  std::vector<std::string> ret;
  for (int i = 0; i < _sz; ++i) {
    ret.push_back(pyobj_string(PyList_GetItem(listObj, i)));
  }
  return ret;
}

#endif
