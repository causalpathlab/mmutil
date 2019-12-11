#include "mmutil_python.hh"

#include "mmutil_python_io.hh"

const char* _read_numpy_desc =
    "Read triplets from a matrix market file and save to a numpy array.\n"
    "\n"
    "Input: A matrix market file name.  The function treats it gzipped\n"
    "       if the file name ends with `.gz`\n"
    "\n"
    "Output: A numpy matrix (C-ordered, row-major)\n";

const char* _read_triplets_desc =
    "Read triplets from a matrix market file and save to a dictionary.\n"
    "\n"
    "Input: A matrix market file name.  The function treats it gzipped\n"
    "       if the file name ends with `.gz`\n"
    "\n"
    "Output: A dictionary with the following keys:\n"
    "       `rows`, `columns`, `values`, `shape`\n";

static PyMethodDef mmutil_methods[] = {
    {"read_triplets", mmutil_read_triplets, METH_VARARGS, _read_triplets_desc},
    {"read_numpy", mmutil_read_numpy, METH_VARARGS, _read_numpy_desc},
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
