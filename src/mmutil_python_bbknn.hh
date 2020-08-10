#include "mmutil_bbknn.hh"
#include "mmutil_python.hh"
#include "mmutil_python_io.hh"

#ifndef MMUTIL_PYTHON_BBKNN_HH_
#define MMUTIL_PYTHON_BBKNN_HH_

const char *_bbknn_desc =
    "\n"
    "[Arguments]\n"
    "mtx_file      : data MTX file (M x N)\n"
    "col_file      : data col file (samples)\n"
    "batch_file    : N x 1 batch assignment file (e.g., individuals) \n"
    "out           : Output file header\n"
    "\n"
    "[Matching options]\n"
    "\n"
    "knn           : k nearest neighbours (default: 1)\n"
    "bilink        : # of bidirectional links (default: 5)\n"
    "nlist         : # nearest neighbor lists (default: 5)\n"
    "\n"
    "rank          : # of SVD factors (default: rank = 50)\n"
    "lu_iter       : # of LU iterations (default: iter = 5)\n"
    "row_weight    : Feature re-weighting (default: none)\n"
    "col_norm      : Column normalization (default: 10000)\n"
    "\n"
    "log_scale     : Data in a log-scale (default: false)\n"
    "raw_scale     : Data in a raw-scale (default: true)\n"
    "\n"
    "[Output]\n"
    "\n"
    "${out}.mtx.gz : (N x N) adjacency matrix\n"
    "                This can be read by `read_triplets`\n"
    "\n";

static PyObject *
mmutil_bbknn(PyObject *self, PyObject *args, PyObject *keywords)
{

    char *mtx_file;
    char *col_file;
    char *batch_file;
    char *out_hdr;

    bbknn_options_t options;

    char *row_weight;

    int knn = options.knn;
    int bilink = options.bilink;
    int nlist = options.nlist;

    int rank = options.rank;
    int lu_iter = options.lu_iter;
    float col_norm = options.col_norm;

    bool log_scale = options.log_scale;
    bool verbose = options.verbose;

    static const char *kwlist[] = { "mtx_file",   //
                                    "col_file",   //
                                    "batch_file", //
                                    "out",        //
                                    "row_weight", //
                                    "knn",        //
                                    "bilink",     //
                                    "nlist",      //
                                    "rank",       //
                                    "lu_iter",    //
                                    "col_norm",   //
                                    "log_scale",  //
                                    "verbose",    //
                                    NULL };

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "ssss|$siiiiifbb",           // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz [s]
                                     &col_file,   // .cols.gz [s]
                                     &batch_file, // .batch.gz [s]
                                     &out_hdr,    // "output" [s]
                                     &row_weight, // a file for row weights [s]
                                     &knn,        // kNN [i]
                                     &bilink,     // bidirectional links [i]
                                     &nlist,      // number of lists [i]
                                     &rank,       // rank of SVD [i]
                                     &lu_iter,    // LU for randomized SVD[i]
                                     &col_norm,   // column-wise normalizer [f]
                                     &log_scale,  // log-scale trans [b]
                                     &verbose)) {
        return NULL;
    }

    options.mtx = std::string(mtx_file);
    options.col = std::string(col_file);
    options.batch = std::string(batch_file);
    options.out = std::string(out_hdr);

    if (row_weight)
        options.row_weight_file = std::string(row_weight);

    if (log_scale) {
        TLOG("Take log-transformed data");
        options.log_scale = true;
        options.raw_scale = false;
    } else {
        options.log_scale = false;
        options.raw_scale = true;
    }

    if (verbose) {
        options.verbose = true;
    }

    options.knn = knn;
    options.bilink = bilink;
    options.nlist = nlist;
    options.rank = rank;
    options.lu_iter = lu_iter;
    options.col_norm = col_norm;

    TLOG("Start building BBKNN ...")
    Py_CHECK(build_bbknn(options));
    TLOG("DONE")

    PyObject *ret = PyDict_New();
    return ret;
}

#endif
