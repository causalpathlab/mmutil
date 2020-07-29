#include "mmutil.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "utils/bgzstream.hh"
#include "mmutil_python.hh"

#ifndef MMUTIL_PYTHON_INDEX_HH_
#define MMUTIL_PYTHON_INDEX_HH_

const char *_build_index_desc =
    "Build an auxiliary column indexing file for fast access for .mtx.gz\n"
    "\n"
    "[Input]\n"
    "\n"
    "mtx_file : .mtx.gz file\n"
    "idx_file : .mtx.gz.index file (optional)\n"
    "\n";

static PyObject *
mmutil_build_index(PyObject *self, PyObject *args, PyObject *keywords)
{

    static const char *kwlist[] = { "mtx_file", "idx_file", NULL };

    char *mtx_file;
    char *_idx_file;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "s|s",                       // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz
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

    if (!is_file_bgz(mtx_file)) {
        convert_bgzip(mtx_file);
    } else {
        TLOG("This file is bgzipped: " << mtx_file);
    }

    Py_CHECK(build_mmutil_index(mtx_file, idx_file));

    PyObject *ret = PyDict_New();
    return ret;
}

#endif
