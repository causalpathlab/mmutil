#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <string>
#include <vector>
#include "utils/io.hh"
#include "utils/util.hh"
#include <numpy/ndarrayobject.h>

#ifndef MMUTIL_PYTHON_HH_
#define MMUTIL_PYTHON_HH_

using Index = long int;
using Scalar = float;
using Triplet = std::tuple<Index, Index, Scalar>;
using TripletVec = std::vector<Triplet>;

#endif
