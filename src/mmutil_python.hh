#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarrayobject.h>
#include <string>
#include <vector>
#include "utils/io.hh"
#include "utils/util.hh"
#include "mmutil.hh"

#ifndef MMUTIL_PYTHON_HH_
#define MMUTIL_PYTHON_HH_

// Already declared in mmutil.hh:
// using Index = size_t;
// using Scalar = float;

using Triplet = std::tuple<Index, Index, Scalar>;
using TripletVec = std::vector<Triplet>;

auto make_argv(const PyObject* args) {
  const Index argc = PyTuple_GET_SIZE(args);
  std::vector<PyObject*> argv(argc);

  for (Index i = 0; i < argc; ++i) {
    argv[i] = PyTuple_GET_ITEM(args, i);
  }

  return argv;
}

#endif
