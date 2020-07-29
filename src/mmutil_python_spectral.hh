#include "mmutil_python.hh"
#include "mmutil_spectral.hh"

#ifndef MMUTIL_PYTHON_SPECTRAL_HH_
#define MMUTIL_PYTHON_SPECTRAL_HH_

const char *_take_svd_desc =
    "Take the eigen spectrum of regularized graph Laplacian.\n"
    "\n"
    "L = S^{-1/2} * A * S^{-1/2}\n"
    "A = X' X\n"
    "S = degree + tau I\n"
    "\n"
    "This function will identify the SVD of Y, where\n"
    "Y = S^{-1/2} * X' and Y = U D V'\n"
    "\n"
    "[Input]\n"
    "file       : A matrix market file name.  The function treats it gzipped\n"
    "             if the file name ends with `.gz`\n"
    "rank       : Set the maximal rank to save computational cost.\n"
    "tau        : Regularization parameter. We add tau/mean_degree * I.\n"
    "iterations : Number of iterations to improve the accuracy of randomized "
    "SVD\n"
    "\n"
    "[Output]\n"
    "A dictionary `out` that contains\n"
    "`out['u']` : U matrix \n"
    "`out['v']` : V matrix \n"
    "`out['d']` : D matrix/vector \n"
    "\n";

static PyObject *
mmutil_take_svd(PyObject *self, PyObject *args, PyObject *keywords);

////////////////////
// implementation //
////////////////////

static PyObject *
mmutil_take_svd(PyObject *self, PyObject *args, PyObject *keywords)
{
    static const char *kwlist[] = { "file", "rank", "tau", "iter", NULL };

    char *mtx_file;
    int rank;
    float tau_scale = 1.0;
    int iterations = 5;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "si|fi",                     // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // filename
                                     &rank,                       //
                                     &tau_scale, // regularization
                                     &iterations // iteration
                                     )) {
        return NULL;
    }

    TLOG("Reading            " << mtx_file);
    TLOG("Regularization:    " << tau_scale);
    TLOG("Rank:              " << rank);
    TLOG("RandAlg iteration: " << iterations);

    eigen_triplet_reader_t::TripletVec Tvec;
    Index max_row, max_col;
    std::tie(Tvec, max_row, max_col) = read_eigen_matrix_market_file(mtx_file);

    TLOG(max_row << " x " << max_col);

    if (max_row < 1 || max_col < 1 || Tvec.size() < 1) {
        PyErr_SetString(PyExc_TypeError, "shouldn't be empty data");
        return NULL;
    }

    SpMat X0(max_row, max_col);
    X0.reserve(Tvec.size());
    X0.setFromTriplets(Tvec.begin(), Tvec.end());

    Mat _U, _V, _D;
    std::tie(_U, _V, _D) =
        take_spectrum_laplacian(X0, tau_scale, rank, iterations);

    TLOG("Output results");

    npy_intp _dims_u[2] = { _U.rows(), _U.cols() };
    PyObject *U = PyArray_ZEROS(2, _dims_u, NPY_FLOAT, NPY_CORDER);
    Scalar *u_data = (Scalar *)PyArray_DATA(U);
    std::copy(_U.data(), _U.data() + _U.size(), u_data);

    npy_intp _dims_v[2] = { _V.rows(), _V.cols() };
    PyObject *V = PyArray_ZEROS(2, _dims_v, NPY_FLOAT, NPY_CORDER);
    Scalar *v_data = (Scalar *)PyArray_DATA(V);
    std::copy(_V.data(), _V.data() + _V.size(), v_data);

    npy_intp _dims_d[2] = { _D.rows(), _D.cols() };
    PyObject *D = PyArray_ZEROS(2, _dims_d, NPY_FLOAT, NPY_CORDER);
    Scalar *d_data = (Scalar *)PyArray_DATA(D);
    std::copy(_D.data(), _D.data() + _D.size(), d_data);

    PyObject *ret = PyDict_New();
    PyObject *_u_key = PyUnicode_FromString("u");
    PyObject *_v_key = PyUnicode_FromString("v");
    PyObject *_d_key = PyUnicode_FromString("d");

    PyDict_SetItem(ret, _u_key, U);
    PyDict_SetItem(ret, _v_key, V);
    PyDict_SetItem(ret, _d_key, D);

    return ret;
}

#endif
