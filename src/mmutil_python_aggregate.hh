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
    "col_norm   : Column normalization (default: 10000)\n"
    "\n"
    "[Output]\n"
    "\n"
    "sum        : (M x n) Sum matrix\n"
    "mean       : (M x n) Mean matrix\n"
    "mu         : (M x n) Depth-adjusted mean matrix\n"
    "mu_sd      : (M x n) SD of the mu matrix\n"
    "cols       : (n x 1) Column names\n"
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

    float col_norm = options.col_norm;
    bool normalize = options.normalize;
    bool discretize = options.discretize;
    bool verbose = options.verbose;

    static const char *kwlist[] = { "mtx_file",        //
                                    "annot_prob_file", //
                                    "ind_file",        //
                                    "lab_file",        //
                                    "out",             //
                                    "discretize",      //
                                    "normalize",       //
                                    "col_norm",        //
                                    "verbose",         //
                                    NULL };

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "sssss|$bbfb",               // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz [s]
                                     &annot_prob_file, // .annot_prob.gz [s]
                                     &ind_file,        // .ind.gz [s]
                                     &lab_file,        // .lab.gz [s]
                                     &out_hdr,         // "output" [s]
                                     &discretize, // discretize Z matrix [b]
                                     &normalize,  // normalize columns [b]
                                     &col_norm,   // column-wise normalizer [f]
                                     &verbose)) {
        return NULL;
    }

    options.mtx_file = std::string(mtx_file);
    options.annot_prob_file = std::string(annot_prob_file);
    options.ind_file = std::string(ind_file);
    options.lab_file = std::string(lab_file);
    options.out = std::string(out_hdr);

    if (verbose) {
        options.verbose = true;
    }

    if (discretize) {
        options.discretize = true;
    } else {
        options.discretize = false;
    }

    if (normalize) {
        options.normalize = true;
    } else {
        options.normalize = false;
    }

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
        Mat _mu;
        Py_CHECK(read_data_file(options.out + ".mu.gz", _mu));
        PyObject *Mu = make_np_array(_mu);
        PyDict_SetItem(ret, PyUnicode_FromString("mu"), Mu);
    }

    {
        Mat _mu_sd;
        Py_CHECK(read_data_file(options.out + ".mu_sd.gz", _mu_sd));
        PyObject *Mu_Sd = make_np_array(_mu_sd);
        PyDict_SetItem(ret, PyUnicode_FromString("mu_sd"), Mu_Sd);
    }

    {
        Mat _sum;
        Py_CHECK(read_data_file(options.out + ".sum.gz", _sum));
        PyObject *Sum = make_np_array(_sum);
        PyDict_SetItem(ret, PyUnicode_FromString("sum"), Sum);
    }

    {
        Mat _sd;
        Py_CHECK(read_data_file(options.out + ".sd.gz", _sd));
        PyObject *Sd = make_np_array(_sd);
        PyDict_SetItem(ret, PyUnicode_FromString("sd"), Sd);
    }

    return ret;
}

#endif
