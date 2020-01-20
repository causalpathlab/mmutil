#include "mmutil_python.hh"

#ifndef MMUTIL_PYTHON_IO_HH_
#define MMUTIL_PYTHON_IO_HH_

const char* _read_triplets_numpy_desc =
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

const char* _write_numpy_desc =
    "Write a numpy array to a file.\n"
    "\n"
    "Input: A numpy matrix (C-ordered, row-major)\n"
    "\n"
    "Output: A (dense) matrix file name.\n"
    "The function treats it gzipped if the name ends with `.gz`\n";

static PyObject*
mmutil_read_triplets_numpy(PyObject* self, PyObject* args);

static PyObject*
mmutil_read_triplets(PyObject* self, PyObject* args);

/////////////////////
// implementations //
/////////////////////

static PyObject*
mmutil_read_triplets_numpy(PyObject* self, PyObject* args) {
  char* _filename;

  if (!PyArg_ParseTuple(args, "s", &_filename)) {
    return NULL;
  }

  TripletVec Tvec;
  Index max_row, max_col;
  const std::string mtx_file(_filename);

  if (!file_exists(mtx_file)) {
    PyErr_SetString(PyExc_TypeError, "file does not exist");
    return NULL;
  }

  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);
  const Index num_elements         = Tvec.size();

  npy_intp dims[2] = {max_row, max_col};

  ////////////////////////////
  // fill data in row-major //
  ////////////////////////////

  TLOG("Allocating a numpy matrix");

  PyObject* ret = PyArray_ZEROS(2, dims, NPY_FLOAT, NPY_CORDER);
  float* data   = (float*)PyArray_DATA(ret);

  const Index INTERVAL = 1e6;
  Index elem           = 0;

  for (auto tt : Tvec) {
    Index i, j;
    float w;
    std::tie(i, j, w) = tt;
    const Index pos   = max_col * i + j;
    data[pos]         = w;
    if ((++elem) % INTERVAL == 0) {
      std::cerr << "\r" << std::setw(30) << "Adding " << std::setw(10)
                << (elem / INTERVAL) << " x 1M triplets (total "
                << std::setw(10) << (num_elements / INTERVAL) << ")"
                << std::flush;
    }
  }
  std::cerr << std::endl;

  TLOG("Return a numpy array with " << elem << " non-zero elements");

  return ret;
}

static PyObject*
mmutil_read_triplets(PyObject* self, PyObject* args) {
  char* _filename;

  if (!PyArg_ParseTuple(args, "s", &_filename)) {
    return NULL;
  }

  TripletVec Tvec;
  Index max_row, max_col;
  const std::string mtx_file(_filename);
  TLOG("Reading " << mtx_file);
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);
  const Index num_elements         = Tvec.size();
  TLOG("Read    " << max_row << " x " << max_col << " with " << num_elements
                  << " elements");

  PyObject* _max_row = PyLong_FromLong(max_row);
  PyObject* _max_col = PyLong_FromLong(max_col);
  PyObject* _shape   = PyList_New(2);
  PyList_SetItem(_shape, 0, _max_row);
  PyList_SetItem(_shape, 1, _max_col);

  if (!_max_row || !_max_col || !_shape) {
    throw std::logic_error("unable to read triplets");
  }

  PyObject* rows   = PyList_New(num_elements);
  PyObject* cols   = PyList_New(num_elements);
  PyObject* values = PyList_New(num_elements);

  const Index INTERVAL = 1e6;
  Index elem           = 0;

  for (auto tt : Tvec) {
    Index i, j;
    Scalar w;
    std::tie(i, j, w) = tt;

    PyObject* ii  = PyLong_FromLong(i);
    PyObject* jj  = PyLong_FromLong(j);
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
      std::cerr << "\r" << std::setw(30) << "Adding " << std::setw(10)
                << (elem / INTERVAL) << " x 1M triplets (total "
                << std::setw(10) << (num_elements / INTERVAL) << ")"
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

  return ret;
}

template <typename OFS, typename T>
void
_write_numpy_array_stream(OFS& ofs, const T* data, const Index nrow,
                          const Index ncol, const Index num_elements) {

  const std::string SEP(" ");
  const Index INTERVAL = 1e6;
  Index elem           = 0;

  for (Index r = 0; r < nrow; ++r) {
    {
      const Index pos = r * ncol;
#ifdef DEBUG
      ASSERT(pos >= 0 && pos < num_elements, "miscalculated #rows and #cols");
#endif
      ofs << data[r * ncol];
    }
    for (Index c = 1; c < ncol; ++c) {
      const Index pos = r * ncol + c;
#ifdef DEBUG
      ASSERT(pos >= 0 && pos < num_elements, "miscalculated #rows and #cols");
#endif
      ofs << SEP << data[pos];

      // if ((++elem) % INTERVAL == 0) {
      //   std::cerr << "\r" << std::setw(30) << "Writing " << std::setw(10) <<
      //   (elem / INTERVAL)
      //             << " x 1M elements (total " << std::setw(10) <<
      //             (num_elements / INTERVAL) <<
      //             ")"
      //             << std::flush;
      // }
    }
    // std::cerr << std::endl;
    ofs << std::endl;
  }
}

template <typename T>
void
_write_numpy_array_file(const std::string _filename, const T* data,
                        const Index nrow, const Index ncol,
                        const Index num_elements) {

  if (file_exists(_filename)) {
    WLOG("File exists: " << _filename);
  }

  if (is_file_gz(_filename)) {
    ogzstream ofs(_filename.c_str(), std::ios::out);
    _write_numpy_array_stream(ofs, data, nrow, ncol, num_elements);
    ofs.close();

  } else {
    std::ofstream ofs(_filename.c_str(), std::ios::out);
    _write_numpy_array_stream(ofs, data, nrow, ncol, num_elements);
    ofs.close();
  }
}

static PyObject*
mmutil_write_numpy(PyObject* self, PyObject* args) {
  PyArrayObject* input = NULL;
  char* _filename      = NULL;

  if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &input, &_filename)) {
    return NULL;
  }

  if (PyArray_NDIM(input) != 2) {
    PyErr_SetString(PyExc_TypeError, "This only supports 2d arrays");
    return NULL;
  }

  npy_intp* dims           = PyArray_DIMS(input);
  const Index nrow         = dims[0];
  const Index ncol         = dims[1];
  const Index num_elements = PyArray_SIZE(input);

  PyArrayObject* input_contig = PyArray_GETCONTIGUOUS((PyArrayObject*)input);

  TLOG("number of elements = " << num_elements);

  const int _npy_type = input->descr->type_num;

  switch (_npy_type) {
    case NPY_DOUBLE:
      _write_numpy_array_file(std::string(_filename),               //
                              (double*)PyArray_DATA(input_contig),  //
                              nrow, ncol,                           //
                              num_elements);                        //
      break;
    case NPY_FLOAT:
      _write_numpy_array_file(std::string(_filename),              //
                              (float*)PyArray_DATA(input_contig),  //
                              nrow, ncol,                          //
                              num_elements);                       //
      break;
    default:
      TLOG("need to implement other data types: \'" <<  //
           input->descr->type << "'");
      break;
  }

  TLOG("Finished");

  Py_XDECREF((PyObject*)input_contig);

  PyObject* ret = PyDict_New();
  return ret;
}

#endif
