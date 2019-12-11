#include "mmutil_python.hh"

#ifndef MMUTIL_PYTHON_IO_HH_
#define MMUTIL_PYTHON_IO_HH_

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

static PyObject* mmutil_read_numpy(PyObject* self, PyObject* args);

static PyObject* mmutil_read_triplets(PyObject* self, PyObject* args);

/////////////////////
// implementations //
/////////////////////

static PyObject* mmutil_read_numpy(PyObject* self, PyObject* args) {
  char* _filename;

  if (!PyArg_ParseTuple(args, "s", &_filename)) {
    return NULL;
  }

  TripletVec Tvec;
  Index max_row, max_col;
  const std::string mtx_file(_filename);
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);
  const Index num_elements = Tvec.size();

  npy_intp dims[2] = {max_row, max_col};

  ////////////////////////////
  // fill data in row-major //
  ////////////////////////////

  TLOG("Allocating a numpy matrix");

  PyObject* ret = PyArray_ZEROS(2, dims, NPY_FLOAT, NPY_CORDER);
  float* data = (float*)PyArray_DATA(ret);

  const Index INTERVAL = 1e6;
  Index elem = 0;

  for (auto tt : Tvec) {
    Index i, j;
    float w;
    std::tie(i, j, w) = tt;
    const Index pos = max_col * i + j;
    data[pos] = w;
    if ((++elem) % INTERVAL == 0) {
      std::cerr << "\r" << std::setw(30) << "Adding " << std::setw(10) << (elem / INTERVAL)
                << " x 1M triplets (total " << std::setw(10) << (num_elements / INTERVAL) << ")"
                << std::flush;
    }
  }
  std::cerr << std::endl;

  TLOG("Return a numpy array with " << elem << " non-zero elements");

  return ret;
}

static PyObject* mmutil_read_triplets(PyObject* self, PyObject* args) {
  char* _filename;

  if (!PyArg_ParseTuple(args, "s", &_filename)) {
    return NULL;
  }

  TripletVec Tvec;
  Index max_row, max_col;
  const std::string mtx_file(_filename);
  TLOG("Reading " << mtx_file);
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);
  const Index num_elements = Tvec.size();
  TLOG("Read    " << max_row << " x " << max_col << " with " << num_elements << " elements");

  PyObject* _max_row = PyLong_FromLong(max_row);
  PyObject* _max_col = PyLong_FromLong(max_col);
  PyObject* _shape = PyList_New(2);
  PyList_SetItem(_shape, 0, _max_row);
  PyList_SetItem(_shape, 1, _max_col);

  if (!_max_row || !_max_col || !_shape) {
    throw std::logic_error("unable to read triplets");
  }

  PyObject* rows = PyList_New(num_elements);
  PyObject* cols = PyList_New(num_elements);
  PyObject* values = PyList_New(num_elements);

  const Index INTERVAL = 1e6;
  Index elem = 0;

  for (auto tt : Tvec) {
    Index i, j;
    Scalar w;
    std::tie(i, j, w) = tt;

    PyObject* ii = PyLong_FromLong(i);
    PyObject* jj = PyLong_FromLong(j);
    PyObject* val = PyFloat_FromDouble(w);

    if (!ii || !jj || !val) {
      Py_DECREF(rows);
      Py_DECREF(cols);
      Py_DECREF(values);
      throw std::logic_error("unable to read triplets");
    }

    PyList_SetItem(rows, elem, ii);
    PyList_SetItem(cols, elem, jj);
    PyList_SetItem(values, elem, val);

    if ((elem + 1) % INTERVAL == 0) {
      std::cerr << "\r" << std::setw(30) << "Adding " << std::setw(10) << (elem / INTERVAL)
                << " x 1M triplets (total " << std::setw(10) << (num_elements / INTERVAL) << ")"
                << std::flush;
    }

    ++elem;
  }

  std::cerr << std::flush;

  PyObject* ret = PyDict_New();

  PyObject* _row_key = PyUnicode_FromString("rows");
  PyObject* _col_key = PyUnicode_FromString("columns");
  PyObject* _val_key = PyUnicode_FromString("values");
  PyObject* _dim_key = PyUnicode_FromString("shape");

  PyDict_SetItem(ret, _row_key, rows);
  PyDict_SetItem(ret, _col_key, cols);
  PyDict_SetItem(ret, _val_key, values);
  PyDict_SetItem(ret, _dim_key, _shape);

  // PyObject* ret = PyList_New(5);
  // PyList_SetItem(ret, 0, rows);
  // PyList_SetItem(ret, 1, cols);
  // PyList_SetItem(ret, 2, values);
  // PyList_SetItem(ret, 3, _max_row);
  // PyList_SetItem(ret, 4, _max_col);
  // TLOG("Return 3 lists (N=" << num_elements << ") with sizes: " << max_row << " x " << max_col);

  return ret;
}

#endif
