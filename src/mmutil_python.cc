#include "mmutil_python.hh"
#include "mmutil_python_io.hh"
#include "mmutil_python_merge.hh"
#include "mmutil_python_annotate.hh"
#include "mmutil_python_util.hh"
#include "mmutil_python_index.hh"
#include "mmutil_python_bbknn.hh"

static PyObject *
mmutil_bbknn(PyObject *self, PyObject *args, PyObject *keywords){



  static const char *kwlist[] = { "mtx_file", "batch_info", "idx_file", NULL };

    char *mtx_file;
    PyObject *batchList;
    char *_idx_file;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "sO!|s",                       // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz
				     &PyList_Type,                // check O!
                                     &batchList,                 // rows
                                     &_idx_file                   // .index
                                     )) {
        return NULL;
    }

    if (!file_exists(mtx_file)) {
        PyErr_SetString(PyExc_TypeError, ".mtx file does not exist");
        return NULL;
    }

    using namespace mmutil::index;

    std::string idx_file;
    if (!_idx_file) {
        idx_file = std::string(mtx_file) + ".index";
    } else {
        idx_file = std::string(_idx_file);
    }

    Py_CHECK(build_mmutil_index(mtx_file, idx_file));



    PyObject *ret = PyDict_New();
    return ret;
}

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
      _build_index_desc       
    },

    { "annotate",
      (PyCFunction)mmutil_annotate,
      METH_VARARGS | METH_KEYWORDS,
      _annotate_desc },

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
