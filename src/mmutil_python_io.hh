#include "mmutil_python.hh"

#ifndef MMUTIL_PYTHON_IO_HH_
#define MMUTIL_PYTHON_IO_HH_

const char *_read_numpy_desc =
    "Read triplets from a matrix market file and save to a numpy array.\n"
    "\n"
    "[Input]\n"
    "mtx_file: Matrix market file name (bgzipped / gzipped)\n"
    "subrow  : a list of row indexes (0-based)\n"
    "subcol  : a list of column indexes (0-based)\n"
    "\n"
    "[Options]\n"
    "idx_file: Indexing file for the matrix file.\n"
    "          Create a new one if it does not exist.\n"
    "\n"
    "[Output]\n"
    "A numpy matrix (C-ordered, row-major)\n";

const char *_read_shape_desc =
    "Take the shape of underlying data matrix.\n"
    "[Input]\n"
    "Matrix marker file name.\n"
    "[Output]\n"
    "A list of [number of rows, number of columns, number of elements]\n"
    "\n";

const char *_read_triplets_desc =
    "Read triplets from a matrix market file and save to a dictionary.\n"
    "\n"
    "Input: A matrix market file name.  The function treats it gzipped\n"
    "       if the file name ends with `.gz`\n"
    "\n"
    "Output: A dictionary with the following keys:\n"
    "       `rows`, `columns`, `values`, `shape`\n";

const char *_write_numpy_desc =
    "Write a numpy array to a file.\n"
    "\n"
    "Input: A numpy matrix (C-ordered, row-major)\n"
    "\n"
    "Output: A (dense) matrix file name.\n"
    "The function treats it gzipped if the name ends with `.gz`\n";

static PyObject *
mmutil_read_shape(PyObject *self, PyObject *args)
{
    char *mtx_file;

    if (!PyArg_ParseTuple(args, "s", &mtx_file)) {
        return NULL;
    }

    using namespace mmutil::bgzf;
    Py_CHECK(convert_bgzip(mtx_file));

    mm_info_reader_t info;                      // fast
    Py_CHECK(peek_bgzf_header(mtx_file, info)); // peaking

    PyObject *_max_row = PyLong_FromLong(info.max_row);
    PyObject *_max_col = PyLong_FromLong(info.max_col);
    PyObject *_max_elem = PyLong_FromLong(info.max_elem);

    PyObject *_shape = PyList_New(3);
    PyList_SetItem(_shape, 0, _max_row);
    PyList_SetItem(_shape, 1, _max_col);
    PyList_SetItem(_shape, 2, _max_elem);

    return _shape;
}

static PyObject *
mmutil_read_numpy(PyObject *self, PyObject *args, PyObject *keywords)
{

    static const char *kwlist[] = { "mtx_file",
                                    "subrow",
                                    "subcol",
                                    "idx_file",
                                    NULL };

    char *mtx_file;
    PyObject *subrowList;
    PyObject *subcolList;
    char *_idx_file;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "sO!O!|s",                   // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz
                                     &PyList_Type,                // check O!
                                     &subrowList,                 // rows
                                     &PyList_Type,                // check O!
                                     &subcolList,                 // columns
                                     &_idx_file                   // .index
                                     )) {
        return NULL;
    }

    if (!file_exists(mtx_file)) {
        PyErr_SetString(PyExc_TypeError, ".mtx file does not exist");
        return NULL;
    }

    using namespace mmutil::index;
    using namespace mmutil::bgzf;

    Py_CHECK(convert_bgzip(mtx_file));

    std::string idx_file;
    if (!_idx_file) {
        idx_file = std::string(mtx_file) + ".index";
    } else {
        idx_file = std::string(_idx_file);
    }

    if (!file_exists(idx_file)) {
        Py_CHECK(build_mmutil_index(mtx_file, idx_file));
    }

    std::vector<Index> idx_tab; // read this should be fast
    Py_CHECK(read_mmutil_index(idx_file, idx_tab));

#ifdef DEBUG
    CHECK(check_index_tab(mtx_file, idx_tab));
#endif

    mm_info_reader_t info;                      // fast
    Py_CHECK(peek_bgzf_header(mtx_file, info)); // peaking
    const Index numFeat = info.max_row;
    const Index N = info.max_col;

    const std::vector<Index> subrow = pyobj_index_vector(subrowList);
    const std::vector<Index> subcol = pyobj_index_vector(subcolList);

    using index_map_t = std::unordered_map<Index, Index>;

    Index max_col = 0;        // Make sure that
    index_map_t subcol_order; // we keep the same order
    for (auto k : subcol) {   // of subcol
        subcol_order[k] = max_col++;
    }

    const auto blocks = find_consecutive_blocks(idx_tab, subcol);

    index_map_t remap_row;
    for (Index new_index = 0; new_index < subrow.size(); ++new_index) {
        const Index old_index = subrow.at(new_index);
        remap_row[old_index] = new_index;
    }

    Index max_row = subrow.size() > 0 ? subrow.size() : info.max_row;

    std::vector<std_triplet_t> Tvec;

    if (subrow.size() > 0) {

        using _reader_t = std_triplet_reader_remapped_rows_cols_t;

        for (auto block : blocks) {
            index_map_t remap_col;
            for (Index old_index = block.lb; old_index < block.ub;
                 ++old_index) {
                remap_col[old_index] = subcol_order[old_index];
            }
            _reader_t reader(Tvec, remap_row, remap_col);
            CHECK(
                visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader));
        }

    } else {

        using _reader_t = std_triplet_reader_remapped_cols_t;

        for (auto block : blocks) {
            index_map_t loc_map;
            for (Index j = block.lb; j < block.ub; ++j) {
                loc_map[j] = subcol_order[j];
            }
            _reader_t reader(Tvec, loc_map);

            CHECK(
                visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader));
        }
    }

    const Index num_elements = Tvec.size();

    npy_intp dims[2] = { max_row, max_col };

    ////////////////////////////
    // fill data in row-major //
    ////////////////////////////

#ifdef DEBUG
    TLOG("Allocating a numpy matrix");
#endif

    PyObject *ret = PyArray_ZEROS(2, dims, NPY_FLOAT, NPY_CORDER);

    float *data = (float *)PyArray_DATA(ret);

    const Index INTERVAL = 1e6;
    Index elem = 0;

    for (auto tt : Tvec) {
        Index i, j;
        float w;
        std::tie(i, j, w) = tt;
        const Index pos = max_col * i + j;
        data[pos] = w;
    }

#ifdef DEBUG
    TLOG("Return a numpy array with " << elem << " non-zero elements");
#endif

    return ret;
}

static PyObject *
_mmutil_read_triplets(const std::string mtx_file)
{

    TripletVec Tvec;
    Index max_row, max_col;

    TLOG("Reading " << mtx_file);
    std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);
    const Index num_elements = Tvec.size();
    TLOG("Read    " << max_row << " x " << max_col << " with " << num_elements
                    << " elements");

    PyObject *_max_row = PyLong_FromLong(max_row);
    PyObject *_max_col = PyLong_FromLong(max_col);
    PyObject *_shape = PyList_New(2);
    PyList_SetItem(_shape, 0, _max_row);
    PyList_SetItem(_shape, 1, _max_col);

    if (!_max_row || !_max_col || !_shape) {
        throw std::logic_error("unable to read triplets");
    }

    PyObject *rows = PyList_New(num_elements);
    PyObject *cols = PyList_New(num_elements);
    PyObject *values = PyList_New(num_elements);

    const Index INTERVAL = 1e6;
    Index elem = 0;

    for (auto tt : Tvec) {
        Index i, j;
        Scalar w;
        std::tie(i, j, w) = tt;

        PyObject *ii = PyLong_FromLong(i);
        PyObject *jj = PyLong_FromLong(j);
        PyObject *val = PyFloat_FromDouble(w);

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

    PyObject *ret = PyDict_New();

    PyObject *_row_key = PyUnicode_FromString("rows");
    PyObject *_col_key = PyUnicode_FromString("columns");
    PyObject *_val_key = PyUnicode_FromString("values");
    PyObject *_dim_key = PyUnicode_FromString("shape");

    PyDict_SetItem(ret, _row_key, rows);
    PyDict_SetItem(ret, _col_key, cols);
    PyDict_SetItem(ret, _val_key, values);
    PyDict_SetItem(ret, _dim_key, _shape);

    return ret;
}

static PyObject *
mmutil_read_triplets(PyObject *self, PyObject *args)
{
    char *_filename;

    if (!PyArg_ParseTuple(args, "s", &_filename)) {
        return NULL;
    }

    return _mmutil_read_triplets(_filename);
}

template <typename OFS, typename T>
void
_write_numpy_array_stream(OFS &ofs,
                          const T *data,
                          const Index nrow,
                          const Index ncol,
                          const Index num_elements)
{
    const std::string SEP(" ");
    const Index INTERVAL = 1e6;
    Index elem = 0;

    for (Index r = 0; r < nrow; ++r) {
        {
            const Index pos = r * ncol;
#ifdef DEBUG
            ASSERT(pos >= 0 && pos < num_elements,
                   "miscalculated #rows and #cols");
#endif
            ofs << data[r * ncol];
        }
        for (Index c = 1; c < ncol; ++c) {
            const Index pos = r * ncol + c;
#ifdef DEBUG
            ASSERT(pos >= 0 && pos < num_elements,
                   "miscalculated #rows and #cols");
#endif
            ofs << SEP << data[pos];

            // if ((++elem) % INTERVAL == 0) {
            //   std::cerr << "\r" << std::setw(30) << "Writing " <<
            //   std::setw(10) << (elem / INTERVAL)
            //             << " x 1M elements (total " <<
            //             std::setw(10) << (num_elements /
            //             INTERVAL) <<
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
_write_numpy_array_file(const std::string _filename,
                        const T *data,
                        const Index nrow,
                        const Index ncol,
                        const Index num_elements)
{
    if (file_exists(_filename)) {
        WLOG("File exists: " << _filename);
    }

    if (is_file_gz(_filename)) {
        obgzf_stream ofs(_filename.c_str(), std::ios::out);
        _write_numpy_array_stream(ofs, data, nrow, ncol, num_elements);
        ofs.close();

    } else {
        std::ofstream ofs(_filename.c_str(), std::ios::out);
        _write_numpy_array_stream(ofs, data, nrow, ncol, num_elements);
        ofs.close();
    }
}

static PyObject *
mmutil_write_numpy(PyObject *self, PyObject *args)
{
    PyArrayObject *input = NULL;
    char *_filename = NULL;

    if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &input, &_filename)) {
        return NULL;
    }

    if (PyArray_NDIM(input) != 2) {
        PyErr_SetString(PyExc_TypeError, "This only supports 2d arrays");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(input);
    const Index nrow = dims[0];
    const Index ncol = dims[1];
    const Index num_elements = PyArray_SIZE(input);

    PyArrayObject *input_contig = PyArray_GETCONTIGUOUS((PyArrayObject *)input);

    TLOG("number of elements = " << num_elements);

    const int _npy_type = input->descr->type_num;

    switch (_npy_type) {
    case NPY_DOUBLE:
        _write_numpy_array_file(std::string(_filename),               //
                                (double *)PyArray_DATA(input_contig), //
                                nrow,
                                ncol,          //
                                num_elements); //
        break;
    case NPY_FLOAT:
        _write_numpy_array_file(std::string(_filename),              //
                                (float *)PyArray_DATA(input_contig), //
                                nrow,
                                ncol,          //
                                num_elements); //
        break;
    default:
        TLOG("need to implement other data types: \'" << //
             input->descr->type << "'");
        break;
    }

    TLOG("Finished");

    Py_XDECREF((PyObject *)input_contig);

    PyObject *ret = PyDict_New();
    return ret;
}

#endif
