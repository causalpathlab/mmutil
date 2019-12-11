#include "mmutil_python.hh"
#include "mmutil_python_io.hh"
#include "mmutil_python_spectral.hh"

static PyMethodDef mmutil_methods[] = {
    {"read_triplets", (PyCFunction)mmutil_read_triplets, METH_VARARGS, _read_triplets_desc},
    {"read_numpy", (PyCFunction)mmutil_read_numpy, METH_VARARGS, _read_numpy_desc},
    {"take_svd", (PyCFunction)mmutil_take_svd, METH_VARARGS | METH_KEYWORDS, _take_svd_desc},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef mmutil_module = {
    PyModuleDef_HEAD_INIT,
    "mmutil",  // name of module
    NULL,      // module documentation
    -1,        // size of per-interpreter
    mmutil_methods,
};

PyMODINIT_FUNC PyInit_mmutil(void) {
  Py_Initialize();
  PyObject* ret = PyModule_Create(&mmutil_module);
  if (ret == NULL) return NULL;
  import_array();
  return ret;
}
