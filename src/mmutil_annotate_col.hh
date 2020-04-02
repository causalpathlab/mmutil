#include <getopt.h>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "eigen_util.hh"
#include "inference/sampler.hh"
#include "io.hh"
#include "mmutil.hh"
#include "mmutil_normalize.hh"
#include "mmutil_embedding.hh"
#include "mmutil_spectral.hh"
#include "mmutil_stat.hh"
#include "utils/progress.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "ext/tabix/bgzf.h"

#ifndef MMUTIL_ANNOTATE_COL_
#define MMUTIL_ANNOTATE_COL_

struct annotation_options_t;
struct annotation_model_t;

int run_annotation(const annotation_options_t &options);

template <typename T>
std::tuple<SpMat, SpMat, std::vector<std::string>, std::vector<std::string>>
read_annotation_matched(const T &options);

template <typename Derived, typename T>
inline std::tuple<std::vector<Index>, std::vector<Index>>
select_rows_columns(const Eigen::SparseMatrixBase<Derived> &ltot,
                    const row_col_stat_collector_t &stat,
                    const T &options);

template <typename Derived, typename T>
void
train_marker_genes(annotation_model_t &annot,
                   const Eigen::MatrixBase<Derived> &_X, // expression matrix
                   const T &options);

/////////////////////////////
// select rows and columns //
/////////////////////////////

template <typename Derived, typename T>
inline std::tuple<std::vector<Index>, std::vector<Index>>
select_rows_columns(const Eigen::SparseMatrixBase<Derived> &ltot,
                    const row_col_stat_collector_t &stat,
                    const T &options)
{
    const Derived &Ltot = ltot.derived();

    auto cv_fun = [](const Scalar &v, const Scalar &m) -> Scalar {
        return std::sqrt(v) / (m + 1e-8);
    };

    std::vector<Index> valid_rows;

    // Select rows by standard deviation
    if (options.balance_marker_size) {
        Vec lab_size = Ltot.transpose() * Mat::Ones(Ltot.rows(), 1);
        const Index sz = lab_size.minCoeff();
        TLOG("Selecting " << sz << " markers for each label");
        {
            const Scalar nn = static_cast<Scalar>(stat.max_col);
            const Scalar mm = std::max(nn - 1.0, 1.0);

            const Vec &s1 = stat.Row_S1;
            const Vec &s2 = stat.Row_S2;

            Vec mu = s1 / nn;
            Vec row_sd = ((s2 - s1.cwiseProduct(mu)) / mm).cwiseSqrt();

            for (Index k = 0; k < Ltot.cols(); ++k) { // each clust
                Vec l_k = row_sd.cwiseProduct(Ltot.col(k));
                auto order = eigen_argsort_descending(l_k);
                std::copy(order.begin(),
                          order.begin() + sz,
                          std::back_inserter(valid_rows));
            }
        }
    } else {
        ///////////////////////////////
        // just select non-zero rows //
        ///////////////////////////////

        Vec col_size = Ltot * Mat::Ones(Ltot.cols(), 1);
        for (Index r = 0; r < col_size.size(); ++r)
            if (col_size(r) > 0)
                valid_rows.push_back(r);
    }

    // Select columns by high coefficient of variance
    std::vector<Index> subcols;
    {
        const Scalar nn = static_cast<Scalar>(stat.max_row);
        const Scalar mm = std::max(nn - 1.0, 1.0);

        const Vec &s1 = stat.Col_S1;
        const Vec &s2 = stat.Col_S2;

        Vec mu = s1 / nn;
        Vec col_cv = ((s2 - s1.cwiseProduct(mu)) / mm).binaryExpr(mu, cv_fun);

        std::vector<Index> index_r = eigen_argsort_descending(col_cv);
        const Index nsubsample = std::min(stat.max_col, options.initial_sample);
        subcols.resize(nsubsample);
        std::copy(index_r.begin(),
                  index_r.begin() + nsubsample,
                  subcols.begin());
    }

    return std::make_tuple(valid_rows, subcols);
}

//////////////////////////////////
// Refine annotation vocabulary //
//////////////////////////////////

struct annotation_model_t {

    explicit annotation_model_t(const Mat lab,
                                const Mat anti_lab,
                                const Scalar _kmax)
        : nmarker(lab.rows())
        , ntype(lab.cols())
        , mu(nmarker, ntype)
        , mu_anti(nmarker, ntype)
        , log_normalizer(ntype)
        , score(ntype)
        , kappa_init(1.0)  //
        , kappa_max(_kmax) //
        , kappa(kappa_init)
        , kappa_anti(kappa_init)
    {
        // initialization
        kappa = kappa_init;
        kappa_anti = kappa_init;
        log_normalizer.setZero();
        mu.setZero();
        mu_anti.setZero();
        mu += lab;
        mu_anti += anti_lab;
    }

    template <typename Derived, typename Derived2, typename Derived3>
    void update_param(const Eigen::MatrixBase<Derived> &_xsum,
                      const Eigen::MatrixBase<Derived2> &_xsum_anti,
                      const Eigen::MatrixBase<Derived3> &_nsum)
    {

        const Derived &Stat = _xsum.derived();
        const Derived2 &Stat_anti = _xsum_anti.derived();
        const Derived3 &nsize = _nsum.derived();

        //////////////////////////////////////////////////
        // concentration parameter for von Mises-Fisher //
        //////////////////////////////////////////////////

        // We use the approximation proposed by Banerjee et al. (2005) for
        // simplicity and relatively stable performance
        //
        //          (rbar*d - rbar^3)
        // kappa = -------------------
        //          1 - rbar^2

        const Scalar d = static_cast<Scalar>(nmarker);

        // We may need to share this kappa estimate across all the
        // types since some of the types might be
        // under-represented.

        const Scalar r = Stat.rowwise().sum().norm() / nsize.sum();

        Scalar _kappa = r * (d - r * r) / (1.0 - r * r);

        if (_kappa > kappa_max) {
            _kappa = kappa_max;
        }

        kappa = _kappa;

        const Scalar r0 = Stat_anti.rowwise().sum().norm() / nsize.sum();

        Scalar _kappa_anti = r0 * (d - r0 * r0) / (1.0 - r0 * r0);

        if (_kappa_anti > kappa_max) {
            _kappa_anti = kappa_max;
        }

        kappa_anti = _kappa_anti;

        ////////////////////////
        // update mean vector //
        ////////////////////////

	mu = Stat * nsize.cwiseInverse().asDiagonal();
        normalize_columns(mu);

	mu_anti = - Stat_anti * nsize.cwiseInverse().asDiagonal();
        normalize_columns(mu_anti);
        // update_log_normalizer();
    }

    template <typename Derived>
    inline const Vec &log_score(const Eigen::MatrixBase<Derived> &_x)
    {
        const Derived &x = _x.derived();
        score = (mu.transpose() * x) * kappa;
        score -= (mu_anti.transpose() * x) * kappa_anti;
        // score += log_normalizer; // not needed
        return score;
    }

    void update_log_normalizer()
    {
        // Normalizer for vMF
        //
        //            kappa^{d/2 -1}
        // C(kappa) = ----------------
        //            (2pi)^{d/2} I(d/2-1, kappa)
        //
        // where
        // I(v,x) = boost::math::cyl_bessel_i(v,x)
        //
        // ln C ~ (d/2 - 1) ln(kappa) - ln I(v, k)
        //

        const Scalar eps = 1e-8;
        const Scalar d = static_cast<Scalar>(nmarker);
        const Scalar df = d * 0.5 - 1.0 + eps;
        const Scalar ln2pi = std::log(2.0 * 3.14159265359);

        auto _log_denom = [&](const Scalar &kap) -> Scalar {
            Scalar ret = (0.5 * d - 1.0) * std::log(kap);
            ret -= ln2pi * (0.5 * d);
            ret -= _log_bessel_i(df, kap);
            return ret;
        };

        log_normalizer.setConstant(_log_denom(kappa));

        // std::cout << "\n\nnormalizer:\n"
        //           << log_normalizer.transpose() << std::endl;
    }

    const Index nmarker;
    const Index ntype;

    Mat mu;             // refined marker x type matrix
    Mat mu_anti;        // permuted marker x type matrix
    Vec log_normalizer; // log-normalizer
    Vec score;          // temporary score

    const Scalar kappa_init;
    const Scalar kappa_max;
    Scalar kappa;
    Scalar kappa_anti;
};

/////////////////////////////
// read matched annotation //
/////////////////////////////

template <typename T>
std::tuple<SpMat, SpMat, std::vector<std::string>, std::vector<std::string>>
read_annotation_matched(const T &options)
{
    using Str = std::string;

    std::vector<std::tuple<Str, Str>> ann_pair_vec;
    if (options.ann.size() > 0) {
        read_pair_file<Str, Str>(options.ann, ann_pair_vec);
    }

    std::vector<std::tuple<Str, Str>> anti_pair_vec;
    if (options.anti_ann.size() > 0) {
        read_pair_file<Str, Str>(options.anti_ann, anti_pair_vec);
    }

    std::vector<Str> row_vec;
    read_vector_file(options.row, row_vec);

    std::unordered_map<Str, Index> row_pos; // row name -> row index
    for (Index j = 0; j < row_vec.size(); ++j) {
        if (row_pos.count(row_vec.at(j)) > 0) {
            WLOG("Duplicate row/feature name: " << row_vec.at(j));
            WLOG("Will ignore the previous one");
        }
        row_pos[row_vec.at(j)] = j;
    }

    std::unordered_map<Str, Index> label_pos; // label name -> label index
    {
        Index j = 0;
        for (auto pp : ann_pair_vec) {
            // ignore missing genes
            if (row_pos.count(std::get<0>(pp)) == 0)
                continue;
            if (label_pos.count(std::get<1>(pp)) == 0)
                label_pos[std::get<1>(pp)] = j++;
        }
    }

    ASSERT(label_pos.size() > 0, "Insufficient #labels");

    using ET = Eigen::Triplet<Scalar>;
    std::vector<ET> triples;

    for (auto pp : ann_pair_vec) {
        if (row_pos.count(std::get<0>(pp)) > 0 &&
            label_pos.count(std::get<1>(pp)) > 0) {
            Index r = row_pos.at(std::get<0>(pp));
            Index l = label_pos.at(std::get<1>(pp));
            triples.push_back(ET(r, l, 1.0));
        }
    }

    std::vector<ET> anti_triples;

    for (auto pp : anti_pair_vec) {
        if (row_pos.count(std::get<0>(pp)) > 0 &&
            label_pos.count(std::get<1>(pp)) > 0) {
            Index r = row_pos.at(std::get<0>(pp));
            Index l = label_pos.at(std::get<1>(pp));
            anti_triples.push_back(ET(r, l, 1.0));
        }
    }

    const Index max_rows = std::max(row_vec.size(), row_pos.size());
    const Index max_labels = label_pos.size();

    SpMat L(max_rows, max_labels);
    L.reserve(triples.size());
    L.setFromTriplets(triples.begin(), triples.end());

    SpMat L0(max_rows, max_labels);
    L0.reserve(anti_triples.size());
    L0.setFromTriplets(anti_triples.begin(), anti_triples.end());

    std::vector<Str> labels(max_labels);
    std::vector<Str> rows(max_rows);

    for (auto pp : label_pos)
        labels[std::get<1>(pp)] = std::get<0>(pp);

    for (auto pp : row_pos)
        rows[std::get<1>(pp)] = std::get<0>(pp);

    if (options.verbose) {
        for (auto l : labels) {
            TLOG("Annotation Labels: " << l);
        }
    }

    return std::make_tuple(L, L0, rows, labels);
}

//////////////////////
// Argument parsing //
//////////////////////

struct annotation_options_t {
    using Str = std::string;

    typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
    const std::vector<Str> METHOD_NAMES;

    annotation_options_t()
    {
        mtx = "";
        col = "";
        row = "";
        ann = "";
        anti_ann = "";
        out = "output.txt.gz";

        col_norm = 10000;

        raw_scale = true;
        log_scale = false;

        initial_sample = 100000;
        batch_size = 100000;

        max_em_iter = 100;

        sampling_method = CV;

        em_tol = 1e-4;
        kappa_max = 100.;

        balance_marker_size = false;
        unconstrained_update = false;
        output_count_matrix = false;

        verbose = false;
    }

    Str mtx;
    Str col;
    Str row;
    Str ann;
    Str anti_ann;
    Str out;

    Scalar col_norm;

    bool raw_scale;
    bool log_scale;

    Index initial_sample;
    Index batch_size;

    Index max_em_iter;

    sampling_method_t sampling_method;

    Scalar em_tol;

    void set_sampling_method(const std::string _method)
    {
        for (int j = 0; j < METHOD_NAMES.size(); ++j) {
            if (METHOD_NAMES.at(j) == _method) {
                sampling_method = static_cast<sampling_method_t>(j);
                break;
            }
        }
    }

    bool balance_marker_size;
    bool unconstrained_update;
    bool output_count_matrix;

    bool verbose;
    Scalar kappa_max;
};

template <typename T>
int
parse_annotation_options(const int argc,     //
                         const char *argv[], //
                         T &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)             : data MTX file\n"
        "--data (-m)            : data MTX file\n"
        "--col (-c)             : data column file\n"
        "--feature (-f)         : data row file (features)\n"
        "--row (-f)             : data row file (features)\n"
        "--ann (-a)             : row annotation file (each line contains a tuple of feature and label)\n"
        "--anti (-A)            : row anti-annotation file  (each line contains a tuple of feature and label)\n"
        "--out (-o)             : Output file header\n"
        "\n"
        "--log_scale (-L)       : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)       : Data in a raw-scale (default: true)\n"
        "\n"
        "--initial_sample (-I)  : Initial sample size (default: 100000)\n"
        "--batch_size (-B)      : Batch size (default: 100000)\n"
        "--kappa_max (-K)       : maximum kappa value (default: 100)\n"
        "MEAN, UNIFORM\n"
        "\n"
        "--em_iter (-i)         : EM iteration (default: 100)\n"
        "--em_tol (-t)          : EM convergence criterion (default: 1e-4)\n"
        "\n"
        "--verbose (-v)         : Set verbose (default: false)\n"
        "--balance_markers (-b) : Balance maker size (default: false)\n"
        "--output_mtx_file (-O) : Write a count matrix of the markers (default: false)\n"
        "\n";

    const char *const short_opts = "m:c:f:a:A:o:I:B:K:M:LRi:t:hbd:u:r:l:kOv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },            //
          { "data", required_argument, nullptr, 'm' },           //
          { "col", required_argument, nullptr, 'c' },            //
          { "row", required_argument, nullptr, 'f' },            //
          { "feature", required_argument, nullptr, 'f' },        //
          { "ann", required_argument, nullptr, 'a' },            //
          { "anti", required_argument, nullptr, 'A' },           //
          { "out", required_argument, nullptr, 'o' },            //
          { "log_scale", no_argument, nullptr, 'L' },            //
          { "raw_scale", no_argument, nullptr, 'R' },            //
          { "initial_sample", required_argument, nullptr, 'I' }, //
          { "batch_size", required_argument, nullptr, 'B' },     //
          { "kappa_max", required_argument, nullptr, 'K' },      //
          { "em_iter", required_argument, nullptr, 'i' },        //
          { "em_tol", required_argument, nullptr, 't' },         //
          { "help", no_argument, nullptr, 'h' },                 //
          { "balance_marker", no_argument, nullptr, 'b' },       //
          { "balance_markers", no_argument, nullptr, 'b' },      //
          { "unconstrained_update", no_argument, nullptr, 'd' }, //
          { "unconstrained", no_argument, nullptr, 'd' },        //
          { "output_mtx_file", no_argument, nullptr, 'O' },      //
          { "verbose", no_argument, nullptr, 'v' },              //
          { nullptr, no_argument, nullptr, 0 } };

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'm':
            options.mtx = std::string(optarg);
            break;

        case 'c':
            options.col = std::string(optarg);
            break;

        case 'f':
            options.row = std::string(optarg);
            break;

        case 'a':
            options.ann = std::string(optarg);
            break;

        case 'A':
            options.anti_ann = std::string(optarg);
            break;

        case 'o':
            options.out = std::string(optarg);
            break;

        case 'i':
            options.max_em_iter = std::stoi(optarg);
            break;

        case 't':
            options.em_tol = std::stof(optarg);
            break;

        case 'I':
            options.initial_sample = std::stoi(optarg);
            break;
        case 'B':
            options.batch_size = std::stoi(optarg);
            break;
        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;
        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;
        case 'M':
            options.set_sampling_method(std::string(optarg));
            break;
        case 'v': // -v or --verbose
            options.verbose = true;
            break;
        case 'b': // -k or --balance_marker
            options.balance_marker_size = true;
            break;
        case 'd':
            options.unconstrained_update = true;
            break;
        case 'K':
            options.kappa_max = std::stof(optarg);
            break;
        case 'O':
            options.output_count_matrix = true;
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.mtx), "No MTX data file");
    ERR_RET(!file_exists(options.col), "No COL data file");
    ERR_RET(!file_exists(options.row), "No ROW data file");
    ERR_RET(!file_exists(options.ann), "No ANN data file");

    return EXIT_SUCCESS;
}

int
run_annotation(const annotation_options_t &options)
{
    //////////////////////////////////////////////////////////
    // Read the annotation information to construct initial //
    // type-specific marker gene profiles                   //
    //////////////////////////////////////////////////////////

    SpMat L_fg, L_bg; // gene x label

    std::vector<std::string> rows;
    std::vector<std::string> labels;

    std::tie(L_fg, L_bg, rows, labels) = read_annotation_matched(options);

    std::vector<std::string> columns;
    CHK_ERR_RET(read_vector_file(options.col, columns),
                "Failed to read the column file: " << options.col);

    CHK_ERR_RET(convert_bgzip(options.mtx),
                "Failed to obtain a bgzipped file: " << options.mtx);

    std::string idx_file = options.mtx + ".index";
    CHK_ERR_RET(build_mmutil_index(options.mtx, idx_file),
                "Failed to construct an index file: " << idx_file);

    std::string mtx_file = options.mtx;

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index N = info.max_col;

    Index batch_size = options.batch_size;

    std::vector<Index> subrow;

    Vec nnz = L_fg * Mat::Ones(L_fg.cols(), 1);
    for (Index r = 0; r < nnz.size(); ++r) {
        if (nnz(r) > 0)
            subrow.emplace_back(r);
    }

    auto log2_op = [](const Scalar &x) -> Scalar { return std::log2(1.0 + x); };

    std::vector<idx_pair_t> idx_tab;
    CHECK(read_mmutil_index(idx_file, idx_tab));

    auto take_batch_data = [&](Index lb, Index ub) -> Mat {
        std::vector<Index> subcol(ub - lb);
        std::iota(subcol.begin(), subcol.end(), lb);
        SpMat x =
            read_eigen_sparse_subset_row_col(mtx_file, idx_tab, subrow, subcol);

        if (options.log_scale) {
            x = x.unaryExpr(log2_op);
        }

        Mat xx = Mat(x);
        normalize_columns(xx);
        return xx;
    };

    Mat L = row_sub(L_fg, subrow);
    Mat L0 = row_sub(L_bg, subrow);

    annotation_model_t annot(L, L0, options.kappa_max);

    ///////////////////////////////
    // Initial greedy assignment //
    ///////////////////////////////

    const Index M = L.rows();
    const Index K = L.cols();
    std::vector<Index> membership(N);
    std::fill(membership.begin(), membership.end(), 0);

    Vec xj(M);
    Vec nsize(K);
    Vec mass(K);

    const Scalar pseudo = 1e-4;
    Mat Stat(M, K);      // feature x label
    Mat Stat_anti(M, K); // feature x label
    nsize.setConstant(pseudo);
    Stat.setConstant(pseudo);      // sum x(g,j) * z(j, k)
    Stat_anti.setConstant(pseudo); // sum x0(g,j) * z(j, k)

    const Index max_em_iter = options.max_em_iter;
    Vec score_trace(max_em_iter);
    Scalar score_init = 0;

    Mat mu = L;
    normalize_columns(mu);

    std::unordered_set<Index> taboo;

    for (Index lb = 0; lb < N; lb += batch_size) {
        const Index ub = std::min(N, batch_size + lb);

        Mat xx = take_batch_data(lb, ub);

        for (Index j = 0; j < xx.cols(); ++j) {
            const Index i = j + lb;

            xj = xx.col(j);
            Index argmax = 0;
            const Vec &_score = annot.log_score(xj);
            _score.maxCoeff(&argmax);
            score_init += _score(argmax);

            if (xj.sum() > 0) {
                nsize(argmax) += 1.0;
                Stat.col(argmax) += xj.cwiseProduct(L.col(argmax));
                Stat_anti.col(argmax) += xj.cwiseProduct(L0.col(argmax));
                membership[i] = argmax;
            } else {
                taboo.insert(i);
            }
        }

        if (options.verbose) {
            std::cerr << nsize.transpose() << "\r" << std::flush;
        }
    }

    score_init /= static_cast<Scalar>(N);
    annot.update_param(Stat, Stat_anti, nsize);

    TLOG("Finished greedy initialization");
    TLOG("Found " << taboo.size() << " cells with no information");

    ////////////////////////////
    // Memoized online update //
    ////////////////////////////

    std::vector<Scalar> em_score_out;

    auto monte_carlo_update = [&]() {
        using DS = discrete_sampler_t<Scalar, Index>;
        DS sampler_k(K); // sample discrete from log-mass
        Scalar score = score_init;
        Vec sj(K); // type x 1 score vector

        for (Index iter = 0; iter < max_em_iter; ++iter) {
            score = 0;

            for (Index lb = 0; lb < N; lb += batch_size) {     // batch
                const Index ub = std::min(N, batch_size + lb); //

                Mat xx = take_batch_data(lb, ub);

                for (Index j = 0; j < xx.cols(); ++j) {

                    const Index i = j + lb;

                    if (taboo.count(i) > 0)
                        continue;

                    xj = xx.col(j);

                    const Index k_prev = membership[i];
                    sj = annot.log_score(xj);
                    const Index k_now = sampler_k(sj);
                    score += sj(k_now);

                    if (k_now != k_prev) {

                        nsize(k_prev) -= 1.0;
                        nsize(k_now) += 1.0;

                        Stat.col(k_prev) -= xj.cwiseProduct(L.col(k_prev));
                        Stat.col(k_now) += xj.cwiseProduct(L.col(k_now));

                        Stat_anti.col(k_prev) -=
                            xj.cwiseProduct(L0.col(k_prev));
                        Stat_anti.col(k_now) += xj.cwiseProduct(L0.col(k_now));

                        membership[i] = k_now;
                    }

                    if (options.verbose) {
                        std::cerr << nsize.transpose() << "\r" << std::flush;
                    }
                } // end of data iteration
                annot.update_param(Stat, Stat_anti, nsize);
            } // end of batch iteration

            score = score / static_cast<Scalar>(N);
            score_trace(iter) = score;

            Scalar diff = std::abs(score_trace(iter));

            if (iter > 4) {
                Scalar score_old = score_trace.segment(iter - 3, 2).sum();
                Scalar score_new = score_trace.segment(iter - 1, 2).sum();
                diff = std::abs(score_old - score_new) /
                    (std::abs(score_old) + options.em_tol);
            } else if (iter > 0) {
                diff = std::abs(score_trace(iter - 1) - score_trace(iter)) /
                    (std::abs(score_trace(iter) + options.em_tol));
            }

            TLOG("Iter [" << iter << "] score = " << score
                          << ", diff = " << diff << ", kappa = " << annot.kappa
                          << ", kappa_null = " << annot.kappa_anti);

            if (iter > 4 && diff < options.em_tol) {

                TLOG("Converged < " << options.em_tol);

                for (Index t = 0; t <= iter; ++t) {
                    em_score_out.emplace_back(score_trace(t));
                }
                break;
            }
        } // end of EM iteration
    };

    annotation_model_t null_annot(L, L0, options.kappa_max);

    auto null_update = [&]() {
        Mat null_stat(M, K);      // feature x label
        Mat null_stat_anti(M, K); // feature x label
        Vec null_nsize(K);
        null_stat.setConstant(pseudo);
        null_stat_anti.setConstant(pseudo);
        null_nsize.setConstant(pseudo);

        std::vector<Index> null_membership(N);
        std::fill(null_membership.begin(), null_membership.end(), 0);

        Vec sj(K);    // type x 1 score vector
        sj.setZero(); // just zero mass
        using DS = discrete_sampler_t<Scalar, Index>;
        DS sampler_k(K); // sample discrete from log-mass

        for (Index lb = 0; lb < N; lb += batch_size) {     // batch
            const Index ub = std::min(N, batch_size + lb); //
            Mat xx = take_batch_data(lb, ub);
            for (Index j = 0; j < xx.cols(); ++j) {
                const Index i = j + lb;
                xj = xx.col(j);
                sj.setZero();
                const Index k_rand = sampler_k(sj);
                sj = annot.log_score(xj);
                const Index k = sampler_k(sj);

                // adding null stat applying  different context
                null_stat.col(k) += L.col(k_rand).cwiseProduct(xj);
                null_stat_anti.col(k) += L0.col(k_rand).cwiseProduct(xj);

                null_membership[i] = k;
                null_nsize(k) += 1.0;
            }
        }

        null_annot.update_param(null_stat, null_stat_anti, null_nsize);
    };

    TLOG("Start training marker gene profiles");
    monte_carlo_update();

    TLOG("Training the null models ...");
    null_update();

    std::vector<std::string> markers;
    markers.reserve(subrow.size());
    std::for_each(subrow.begin(), subrow.end(), [&](const auto r) {
        markers.emplace_back(rows.at(r));
    });
    write_vector_file(options.out + ".marker_names.gz", markers);
    write_vector_file(options.out + ".label_names.gz", labels);
    write_data_file(options.out + ".marker_profile.gz", annot.mu);
    write_data_file(options.out + ".marker_profile_anti.gz", annot.mu_anti);

    write_vector_file(options.out + ".em_scores.gz", em_score_out);

    //////////////////////////////////////////////
    // Assign labels to all the cells (columns) //
    //////////////////////////////////////////////

    using out_tup = std::tuple<std::string, std::string, Scalar, Scalar>;
    std::vector<out_tup> output;

    output.reserve(N);
    Vec zi(annot.ntype);
    Vec null_zi(annot.ntype);
    Mat Pr(annot.ntype, N);
    Mat null_Pr(annot.ntype, N);

    for (Index lb = 0; lb < N; lb += batch_size) {
        const Index ub = std::min(N, batch_size + lb);

        Mat xx = take_batch_data(lb, ub);

        for (Index j = 0; j < xx.cols(); ++j) {
            const Index i = j + lb;
            const Vec &_score = annot.log_score(xx.col(j));
            normalized_exp(_score, zi);

            Index argmax;
            _score.maxCoeff(&argmax);
            Pr.col(i) = zi;

            const Vec &_score_null = null_annot.log_score(xx.col(j));
            normalized_exp(_score_null, null_zi);
            null_Pr.col(i) = null_zi;

            output.emplace_back(columns.at(i),
                                labels.at(argmax),
                                zi(argmax),
                                null_zi(argmax));
        }
        TLOG("Annotated on the batch [" << lb << ", " << ub << ")");
    }

    Pr.transposeInPlace();
    null_Pr.transposeInPlace();

    write_tuple_file(options.out + ".annot.gz", output);
    write_data_file(options.out + ".annot_prob.gz", Pr);
    write_data_file(options.out + ".null_prob.gz", null_Pr);

    TLOG("Done");
    return EXIT_SUCCESS;
}

#endif
