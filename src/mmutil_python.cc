// Python API for Matrix Market Utility
// (c) Yongjin Park

#include "mmutil_python.hh"
#include "mmutil_python_io.hh"
#include "mmutil_python_merge.hh"
#include "mmutil_python_annotate.hh"
#include "mmutil_python_util.hh"
#include "mmutil_python_index.hh"
#include "mmutil_python_aggregate.hh"
#include "mmutil_python_bbknn.hh"

static PyMethodDef mmutil_methods[] = {
    { "read_triplets",
      (PyCFunction)mmutil_read_triplets,
      METH_VARARGS,
      _read_triplets_desc },

    { "read_shape",
      (PyCFunction)mmutil_read_shape,
      METH_VARARGS,
      _read_shape_desc },

    { "read_numpy",
      (PyCFunction)mmutil_read_numpy,
      METH_VARARGS | METH_KEYWORDS,
      _read_numpy_desc },

    { "write_numpy",
      (PyCFunction)mmutil_write_numpy,
      METH_VARARGS,
      _write_numpy_desc },

    { "index",
      (PyCFunction)mmutil_build_index,
      METH_VARARGS,
      _build_index_desc },

    { "annotate",
      (PyCFunction)mmutil_annotate,
      METH_VARARGS | METH_KEYWORDS,
      _annotate_desc },

    { "aggregate",
      (PyCFunction)mmutil_aggregate,
      METH_VARARGS | METH_KEYWORDS,
      _aggregate_desc },

    { "bbknn",
      (PyCFunction)mmutil_bbknn,
      METH_VARARGS | METH_KEYWORDS,
      _bbknn_desc },

    { "merge",
      (PyCFunction)mmutil_merge_files,
      METH_VARARGS | METH_KEYWORDS,
      _merge_files_desc },

    { "select",
      (PyCFunction)mmutil_select,
      METH_VARARGS | METH_KEYWORDS,
      _select_desc },

    { NULL, NULL, 0, NULL },
};

const char *module_desc = "* read_triplets\n"
                          "* read_shape\n"
                          "* read_numpy\n"
                          "* write_numpy\n"
                          "* merge_files\n"
                          "* annotate\n"
                          "* aggregate\n"
                          "* bbknn\n"
                          "\n";

static struct PyModuleDef mmutil_module = {
    PyModuleDef_HEAD_INIT,
    "mmutil",    // name of module
    module_desc, // module documentation
    -1,          // size of per-interpreter
    mmutil_methods,
};

PyMODINIT_FUNC
PyInit_mmutil(void)
{
    Py_Initialize();
    PyObject *ret = PyModule_Create(&mmutil_module);
    if (ret == NULL)
        return NULL;
    import_array();
    return ret;
}
