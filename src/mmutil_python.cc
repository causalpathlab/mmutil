#include "mmutil_python.hh"

#include "mmutil_python_io.hh"
#include "mmutil_python_merge.hh"
#include "mmutil_python_spectral.hh"

static PyMethodDef mmutil_methods[] = {
    {"read_triplets", (PyCFunction)mmutil_read_triplets, METH_VARARGS,
     _read_triplets_desc},
    {"read_triplets_numpy", (PyCFunction)mmutil_read_triplets_numpy,
     METH_VARARGS, _read_triplets_numpy_desc},
    {"write_numpy", (PyCFunction)mmutil_write_numpy, METH_VARARGS,
     _write_numpy_desc},
    {"take_svd", (PyCFunction)mmutil_take_svd, METH_VARARGS | METH_KEYWORDS,
     _take_svd_desc},
    {"merge_files", (PyCFunction)mmutil_merge_files,
     METH_VARARGS | METH_KEYWORDS, _merge_files_desc},
    {NULL, NULL, 0, NULL},
};

const char* module_desc =
    "* read_triplets\n"
    "* read_triplets_numpy\n"
    "* write_numpy\n"
    "* take_svd\n"
    "* merge_files\n"
    "\n";

static struct PyModuleDef mmutil_module = {
    PyModuleDef_HEAD_INIT,
    "mmutil",     // name of module
    module_desc,  // module documentation
    -1,           // size of per-interpreter
    mmutil_methods,
};

PyMODINIT_FUNC
PyInit_mmutil(void) {
  Py_Initialize();
  PyObject* ret = PyModule_Create(&mmutil_module);
  if (ret == NULL) return NULL;
  import_array();
  return ret;
}
