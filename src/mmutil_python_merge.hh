#include "io.hh"
#include "mmutil_merge_col.hh"
#include "mmutil_python.hh"

#ifndef MMUTIL_PYTHON_MERGE_HH_
#define MMUTIL_PYTHON_MERGE_HH_

const char* _merge_files_desc =
    "Merge the columns of sparse matrices matching rows\n"
    "\n"
    "glob_row_file   : A file that contains the names of rows.\n"
    "mtx_file[i]     : i-th matrix market format file\n"
    "row_file[i]     : i-th row file\n"
    "col_file[i]     : i-th column file\n"
    "output          : Header string for the output fileset.\n"
    "count_threshold : Set the number of non-zero elements per column\n"
    "\n";

static PyObject*
mmutil_merge_files(PyObject* self, PyObject* args, PyObject* keywords) {

  static const char* kwlist[] = {"glob_row_file",  //
                                 "mtx_files",      //
                                 "row_files",      //
                                 "col_files",      //
                                 "output",         //
                                 "threshold",      //
                                 NULL};

  char* _glob_row_file;
  PyObject* mtxList;
  PyObject* rowList;
  PyObject* colList;
  char* _out;
  int column_threshold = 0;  // no thresholding

  if (!PyArg_ParseTupleAndKeywords(args, keywords, "sO!O!O!s|i",  //
                                   const_cast<char**>(kwlist),    //
                                   &_glob_row_file,               //
                                   &PyList_Type, &mtxList,        // O! checks type
                                   &PyList_Type, &rowList,        // O! checks type
                                   &PyList_Type, &colList,        // O! checks type
                                   &_out,                         //
                                   &column_threshold)) {
    return NULL;
  }

  const std::string glob_row_file(_glob_row_file);
  const std::string output(_out);

  TLOG(glob_row_file);
  TLOG(output);

  const std::vector<std::string> mtx_files = pyobj_string_vector(mtxList);
  const std::vector<std::string> row_files = pyobj_string_vector(rowList);
  const std::vector<std::string> col_files = pyobj_string_vector(colList);

  const int num_batches = mtx_files.size();

  PyObject* ret = PyDict_New();

  if (num_batches != row_files.size() && num_batches != col_files.size()) {
    PyErr_SetString(PyExc_TypeError, "all the lists should contain the same number of elements");
    return ret;
  }

  TLOG("Total number of batches: " << num_batches);

  int flag = run_merge_col(glob_row_file,     //
                           column_threshold,  //
                           output,            //
                           mtx_files,         //
                           row_files,         //
                           col_files);

  if (flag != EXIT_SUCCESS) {
    ELOG("Found a problem in merging files");
  }

  TLOG("Finished writing the combined files");
  return ret;
}

#endif
