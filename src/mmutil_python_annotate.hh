#include "mmutil_python.hh"
#include "mmutil_annotate.hh"

#ifndef MMUTIL_PYTHON_ANNOTATE_HH_
#define MMUTIL_PYTHON_ANNOTATE_HH_

const char *_annotate_desc =
    "[Input]\n"
    "\n"
    "mtx_file   : data MTX file\n"
    "col_file   : data column file\n"
    "row_file   : data row file (features)\n"
    "ann_file   : row annotation file; each line contains a tuple of feature and label\n"
    "anti_file  : row anti-annotation file; each line contains a tuple of feature and label\n"
    "qc_file    : row annotation file for Q/C; each line contains a tuple of feature and minimum score\n"
    "out        : Output file header\n"
    "log_scale  : Data in a log-scale (default: false)\n"
    "raw_scale  : Data in a raw-scale (default: true)\n"
    "batch_size : Batch size (default: 100000)\n"
    "kappa_max  : maximum kappa value (default: 100)\n"
    "em_iter    : EM iteration (default: 100)\n"
    "em_tol     : EM convergence criterion (default: 1e-4)\n"
    "verbose    : Set verbose (default: false)\n"
    "\n"
    "[Output]\n"
    "\n"
    "prob       : column x label numpy matrix\n"
    "mu         : marker x label mean matrix\n"
    "mu_anti    : marker x label (anti) mean matrix\n"
    "marker     : a list of marker names\n"
    "label      : a list of label names\n"
    "\n";

static PyObject *
mmutil_annotate(PyObject *self, PyObject *args, PyObject *keywords)
{

    static const char *kwlist[] = { "mtx_file",   //
                                    "row_file",   //
                                    "col_file",   //
                                    "ann_file",   //
                                    "out_hdr",    //
                                    "anti_file",  //
                                    "qc_file",    //
                                    "log_scale",  //
                                    "batch_size", //
                                    "kappa_max",  //
                                    "em_iter",    //
                                    "em_tol",     //
                                    "verbose",    //
                                    NULL };

    char *mtx_file;
    char *row_file;
    char *col_file;
    char *ann_file;
    char *out_hdr;

    char *anti_file;
    char *qc_file;

    annotation_options_t options;

    bool log_scale = options.log_scale;
    int batch_size = options.batch_size;
    float kappa_max = options.kappa_max;
    int em_iter = options.max_em_iter;
    float em_tol = options.em_tol;
    bool verbose = options.verbose;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     keywords,                    // keywords
                                     "sssss|$ssbififb",           // format
                                     const_cast<char **>(kwlist), //
                                     &mtx_file,                   // .mtx.gz
                                     &row_file,                   // .row.gz
                                     &col_file,                   // .cols.gz
                                     &ann_file,                   // annotation
                                     &out_hdr,    // output header
                                     &anti_file,  // anti-
                                     &qc_file,    // Q/C
                                     &log_scale,  // log-scale
                                     &batch_size, // batch size
                                     &kappa_max,  // concentration
                                     &em_iter,    // EM iterations
                                     &em_tol,     // EM convergence
                                     &verbose)) {
        return NULL;
    }

    options.mtx = std::string(mtx_file);
    options.row = std::string(row_file);
    options.col = std::string(col_file);
    options.ann = std::string(ann_file);
    options.out = std::string(out_hdr);

    if (anti_file)
        options.anti_ann = std::string(anti_file);

    if (qc_file)
        options.qc_ann = std::string(qc_file);

    options.max_em_iter = em_iter;
    options.em_tol = em_tol;
    options.batch_size = batch_size;
    options.kappa_max = kappa_max;

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

    Py_CHECK(run_annotation(options)); // Fit von Mises-Fisher

    Mat _pr;
    Py_CHECK(read_data_file(options.out + ".annot_prob.gz", _pr));
    PyObject *Pr = make_np_array(_pr);

    Mat _mu;
    Py_CHECK(read_data_file(options.out + ".marker_profile.gz", _mu));
    PyObject *Mu = make_np_array(_mu);

    Mat _mu_anti;
    Py_CHECK(read_data_file(options.out + ".marker_profile_anti.gz", _mu_anti));
    PyObject *Mu_anti = make_np_array(_mu_anti);

    std::vector<std::string> _lab;
    Py_CHECK(read_vector_file(options.out + ".label_names.gz", _lab));

    std::vector<std::string> _marker;
    Py_CHECK(read_vector_file(options.out + ".marker_names.gz", _marker));

    PyObject *labels = PyList_New(_lab.size());
    for (Index i = 0; i < _lab.size(); ++i)
        PyList_SetItem(labels, i, PyUnicode_FromString(_lab[i].c_str()));

    PyObject *markers = PyList_New(_marker.size());
    for (Index i = 0; i < _marker.size(); ++i)
        PyList_SetItem(markers, i, PyUnicode_FromString(_marker[i].c_str()));

    PyObject *ret = PyDict_New();
    PyDict_SetItem(ret, PyUnicode_FromString("prob"), Pr);
    PyDict_SetItem(ret, PyUnicode_FromString("mu"), Mu);
    PyDict_SetItem(ret, PyUnicode_FromString("mu_anti"), Mu_anti);
    PyDict_SetItem(ret, PyUnicode_FromString("marker"), markers);
    PyDict_SetItem(ret, PyUnicode_FromString("label"), labels);

    return ret;
}

#endif
