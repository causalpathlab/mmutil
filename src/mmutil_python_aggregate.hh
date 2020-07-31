#include "mmutil_python.hh"
#include "mmutil_aggregate_col.hh"

#ifndef MMUTIL_PYTHON_AGGREGATE_HH_
#define MMUTIL_PYTHON_AGGREGATE_HH_

const char *_aggregate_desc =
    "[Input]\n"
    "\n"
    "mtx_file   : data MTX file\n"
    "prob_file  : annotation/clustering probability (N x K)\n"
    "ind_file   : N x 1 sample to individual (n)\n"
    "lab_file   : K x 1 (cell type) annotation label name\n"
    "out        : Output file header\n"
    "\n"
    "verbose    : Set verbose (default: false)\n"
    "\n"
    "[Counterfactual matching options]\n"
    "\n"
    "trt_file   : N x 1 sample to case-control membership\n"
    "knn        : k nearest neighbours (default: 1)\n"
    "bilink     : # of bidirectional links (default: 5)\n"
    "nlist      : # nearest neighbor lists (default: 5)\n"
    "\n"
    "rank       : # of SVD factors (default: rank = 50)\n"
    "lu_iter    : # of LU iterations (default: iter = 5)\n"
    "row_weight : Feature re-weighting (default: none)\n"
    "col_norm   : Column normalization (default: 10000)\n"
    "\n"
    "log_scale  : Data in a log-scale (default: false)\n"
    "\n"
    "[Inference Options]\n"
    "\n"
    "gibbs      : number of gibbs sampling (default: 100)\n"
    "burnin     : number of burn-in sampling (default: 10)\n"
    "\n"
    "[Output]\n"
    "\n"
    "mean       : (M x n) Mean matrix\n"
    "sd         : (M x n) SD matrix\n"
    "cols       : (n x 1) Column names\n"
    "mean_cf    : (M x n) Counterfactual mean matrix\n"
    "sd_cf      : (M x n) Counterfactual sd matrix\n"
    "\n";

static PyObject *
mmutil_aggregate(PyObject *self, PyObject *args, PyObject *keywords)
{
    char *mtx_file;
    char *annot_prob_file;
    char *ind_file;
    char *lab_file;
    char *out_hdr;

    aggregate_options_t options;

    char *trt_file;
    char *row_weight;

    int gibbs = options.ngibbs;
    int burnin = options.nburnin;
    int knn = options.knn;
    int bilink = options.bilink;
    int nlist = options.nlist;
    int rank = options.rank;
    int lu_iter = options.lu_iter;
    float col_norm = options.col_norm;
    bool log_scale = options.log_scale;
    bool discretize = options.discretize;
    bool verbose = options.verbose;

    static const char *kwlist[] = { "mtx_file",        //
                                    "annot_prob_file", //
                                    "ind_file",        //
                                    "lab_file",        //
                                    "out",             //
                                    "trt_file",        //
                                    "row_weight",      //
                                    "gibbs",           //
                                    "burnin",          //
                                    "knn",             //
                                    "bilink",          //
                                    "nlist",           //
                                    "rank",            //
                                    "lu_iter",         //
                                    "col_norm",        //
                                    "log_scale",       //
                                    "discretize",      //
                                    "verbose",         //
                                    NULL };

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "sssss|$ssiiiiiiifbbb",      // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz [s]
                                     &annot_prob_file, // .annot_prob.gz [s]
                                     &ind_file,        // .ind.gz [s]
                                     &lab_file,        // .lab.gz [s]
                                     &out_hdr,         // "output" [s]
                                     &trt_file,   // treatment file [s] -------
                                     &row_weight, // a file for row weights [s]
                                     &gibbs,      // gibbs sampling [i]
                                     &burnin,     // burn-in iterations [i]
                                     &knn,        // kNN [i]
                                     &bilink,     // bidirectional links [i]
                                     &nlist,      // number of lists [i]
                                     &rank,       // rank of SVD [i]
                                     &lu_iter,    // LU for randomized SVD[i]
                                     &col_norm,   // column-wise normalizer [f]
                                     &log_scale,  // log-scale trans [b]
                                     &discretize, // discretize Z matrix [b]
                                     &verbose)) {
        return NULL;
    }

    options.mtx = std::string(mtx_file);
    options.annot_prob = std::string(annot_prob_file);
    options.ind = std::string(ind_file);
    options.lab = std::string(lab_file);
    options.out = std::string(out_hdr);

    if (trt_file)
        options.trt_ind = std::string(trt_file);

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

    if (discretize) {
        options.discretize = true;
    } else {
        options.discretize = false;
    }

    options.ngibbs = gibbs;
    options.nburnin = burnin;
    options.knn = knn;
    options.bilink = bilink;
    options.nlist = nlist;
    options.rank = rank;
    options.lu_iter = lu_iter;
    options.col_norm = col_norm;

    TLOG("Start aggregating...")
    Py_CHECK(aggregate_col(options));
    TLOG("Done");

    PyObject *ret = PyDict_New();

    std::vector<std::string> _col;
    Py_CHECK(read_vector_file(options.out + ".cols.gz", _col));

    PyObject *cols = PyList_New(_col.size());
    for (Index i = 0; i < _col.size(); ++i)
        PyList_SetItem(cols, i, PyUnicode_FromString(_col[i].c_str()));

    PyDict_SetItem(ret, PyUnicode_FromString("cols"), cols);

    {
        Mat _mean;
        Py_CHECK(read_data_file(options.out + ".mean.gz", _mean));
        PyObject *Mean = make_np_array(_mean);
        PyDict_SetItem(ret, PyUnicode_FromString("mean"), Mean);
    }

    {
        Mat _sd;
        Py_CHECK(read_data_file(options.out + ".sd.gz", _sd));
        PyObject *Sd = make_np_array(_sd);
        PyDict_SetItem(ret, PyUnicode_FromString("sd"), Sd);
    }

    if (file_exists(options.out + ".cf_mean.gz")) {
        Mat _mean;
        Py_CHECK(read_data_file(options.out + ".mean.gz", _mean));
        PyObject *Mean = make_np_array(_mean);
        PyDict_SetItem(ret, PyUnicode_FromString("cf_mean"), Mean);
    }

    if (file_exists(options.out + ".cf_sd.gz")) {
        Mat _sd;
        Py_CHECK(read_data_file(options.out + ".sd.gz", _sd));
        PyObject *Sd = make_np_array(_sd);
        PyDict_SetItem(ret, PyUnicode_FromString("cf_sd"), Sd);
    }

    return ret;
}

#endif
