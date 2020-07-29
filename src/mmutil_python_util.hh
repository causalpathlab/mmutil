#include "mmutil_python.hh"
#include "mmutil_select.hh"

#ifndef MMUTIL_PYTHON_UTIL_HH_
#define MMUTIL_PYTHON_UTIL_HH_

const char *_select_desc =
    "Create a new fileset selecting rows or columns\n"
    "\n"
    "[Input]\n"
    "mtx_file : Matrix market file name (bgzipped / gzipped)\n"
    "row_file : Row file name (gzipped)\n"
    "col_file : Column file name (gzipped)\n"
    "out_hdr  : Header for output files\n"
    "subrow   : a list of row names\n"
    "subcol   : a list of column names\n"
    "\n";

static PyObject *
mmutil_select(PyObject *self, PyObject *args, PyObject *keywords)
{

    static const char *kwlist[] = { "mtx_file", //
                                    "row_file", //
                                    "col_file", //
                                    "out_hdr",  //
                                    "subrow",   //
                                    "subcol",   //
                                    NULL };

    char *mtx_file;
    char *row_file;
    char *col_file;
    char *out_hdr;

    PyObject *subrowList;
    PyObject *subcolList;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "ssss|O!O!",                // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz
                                     &row_file,                   // .row.gz
                                     &col_file,                   // .cols.gz
                                     &out_hdr,     // output header
                                     &PyList_Type, // check O!
                                     &subrowList,  // rows
                                     &PyList_Type, // check O!
                                     &subcolList   // columns
                                     )) {
        return NULL;
    }

    if (!file_exists(mtx_file)) {
        PyErr_SetString(PyExc_TypeError, ".mtx file does not exist");
        return NULL;
    }

    if (!file_exists(row_file)) {
        PyErr_SetString(PyExc_TypeError, ".row file does not exist");
        return NULL;
    }

    if (!file_exists(col_file)) {
        PyErr_SetString(PyExc_TypeError, ".col file does not exist");
        return NULL;
    }

    PyObject *ret = PyDict_New();

    if (subcolList) {

        const std::vector<std::string> subcol = pyobj_string_vector(subcolList);
        Py_CHECK(copy_selected_columns(mtx_file, col_file, subcol, out_hdr));

    } else if (subrowList) {

        const std::vector<std::string> subrow = pyobj_string_vector(subrowList);
        Py_CHECK(copy_selected_rows(mtx_file, row_file, subrow, out_hdr));

    } else {
        TLOG("Provide a list of row or column names");
    }

    return ret;
}

#endif
