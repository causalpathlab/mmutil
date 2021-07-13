#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "mmutil.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "mmutil_util.hh"
#include "mmutil_stat.hh"
#include "mmutil_bgzf_util.hh"

#include "utils/util.hh"
#include "utils/sse.h"
#include "utils/cast.h"
#include "utils/math.hh"
#include "utils/fastexp.h"
#include "utils/fastlog.h"
#include "utils/fastgamma.h"
#include "utils/std_util.hh"
#include "utils/eigen_util.hh"
#include "io_visitor.hh"
#include "utils/gzstream.hh"
#include "utils/util.hh"
#include "utils/check.hh"
#include "utils/tuple_util.hh"
#include "utils/progress.hh"
#include "utils/strbuf.hh"
#include "utils/bgzstream.hh"

#ifdef __cplusplus
extern "C" {
#endif

#include "ext/tabix/bgzf.h"
#include "ext/tabix/kstring.h"

#ifdef __cplusplus
}
#endif

#ifndef MMUTIL_PYTHON_HH_
#define MMUTIL_PYTHON_HH_

using namespace mmutil::index;
using namespace mmutil::io;

#define Py_CHECK(cond)                \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            return NULL;              \
        }                             \
    }

using Triplet = std::tuple<Index, Index, Scalar>;
using TripletVec = std::vector<Triplet>;

auto
make_argv(const PyObject *args)
{
    const Index argc = PyTuple_GET_SIZE(args);
    std::vector<PyObject *> argv(argc);

    for (Index i = 0; i < argc; ++i) {
        argv[i] = PyTuple_GET_ITEM(args, i);
    }

    return argv;
}

inline std::string
pyobj_string(PyObject *obj)
{
    return PyBytes_AsString(PyUnicode_AsEncodedString(obj, "UTF-8", "strict"));
}

inline std::vector<std::string>
pyobj_string_vector(PyObject *listObj)
{
    const int _sz = PyList_Size(listObj);
    std::vector<std::string> ret;
    for (int i = 0; i < _sz; ++i) {
        ret.push_back(pyobj_string(PyList_GetItem(listObj, i)));
    }
    return ret;
}

inline std::vector<Index>
pyobj_index_vector(PyObject *listObj)
{
    const int _sz = PyList_Size(listObj);
    std::vector<Index> ret;
    for (int i = 0; i < _sz; ++i) {
        ret.push_back(PyLong_AsLongLong(PyList_GetItem(listObj, i)));
    }
    return ret;
}

inline PyObject *
make_np_array(Mat &_A)
{
    npy_intp _dims_a[2] = { _A.rows(), _A.cols() };
    PyObject *A = PyArray_ZEROS(2, _dims_a, NPY_FLOAT, NPY_CORDER);
    Scalar *a_data = (Scalar *)PyArray_DATA(A);
    std::copy(_A.data(), _A.data() + _A.size(), a_data);
    return A;
}

#endif
