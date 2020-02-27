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
#include "mmutil_spectral.hh"
#include "mmutil_stat.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_ANNOTATE_COL_
#define MMUTIL_ANNOTATE_COL_

struct annotate_options_t;

template <typename T>
std::tuple<SpMat, std::vector<std::string>, std::vector<std::string>>
read_annotation_matched(const T &options);

template <typename Derived, typename T>
inline std::tuple<std::vector<Index>, std::vector<Index>>
select_rows_columns(const Eigen::SparseMatrixBase<Derived> &ltot,
                    const row_col_stat_collector_t &stat,
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

struct annot_model_t {

    explicit annot_model_t(const Mat lab)
        : L(lab)
        , nmarker(L.rows())
        , ntype(L.cols())
        , mu(nmarker, ntype)
        , mu_null(nmarker, ntype)
        , kappa(ntype)
        , rbar(ntype)
        , log_normalizer(ntype)
        , score(ntype)
        , kappa_init(1.0)
    {
        // initialization
        kappa.setConstant(kappa_init);
        log_normalizer.setZero();
        mu = L;
    }

    template <typename Derived, typename Derived2>
    void update_param(const Eigen::MatrixBase<Derived> &_xsum,
                      const Eigen::MatrixBase<Derived2> &_nsum)
    {

        const Derived &Stat = _xsum.derived();
        const Derived2 &nsize = _nsum.derived();

        //////////////////////////////////////////////////
        // concentration parameter for von Mises-Fisher //
        //////////////////////////////////////////////////

        // We use the approximation proposed by Banerjee et al. (2005) for
        // simplicity and relatively stable performance
        //
        //          (rbar*d - rbar^3)
        // kappa = -------------------
        //          1 - rbar^2

        // We need to share this kappa estimate across all the types
        // since some of the types might be under-represented.

        const Scalar r = Stat.rowwise().sum().norm() / nsize.sum();
        rbar.setConstant(r);

        const Scalar d = static_cast<Scalar>(nmarker);
        const Scalar eps = 1e-8;

        const Scalar _kappa = r * (d - r * r) / (1.0 - r * r);

        kappa.setConstant(_kappa);

        ////////////////////////
        // update mean vector //
        ////////////////////////

        mu = Stat * nsize.cwiseInverse().asDiagonal();
        normalize_columns(mu);
        update_log_normalizer();
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

        log_normalizer = kappa.unaryExpr(_log_denom);

        // std::cout << "\n\nnormalizer:\n"
        //           << log_normalizer.transpose() << std::endl;
    }

    template <typename Derived>
    inline const Vec &log_score(const Eigen::MatrixBase<Derived> &_x)
    {
        const Derived &x = _x.derived();
        score = (mu.transpose() * x).cwiseProduct(kappa);
        score += log_normalizer; // not needed
        return score;
    }

    const Mat L;
    const Index nmarker;
    const Index ntype;

    Mat mu; // refined marker x type matrix
    Mat mu_null; // permuted marker x type matrix
    Vec kappa; // von Mises-Fisher concentration
    Vec rbar; // temporary
    Vec log_normalizer; // log-normalizer
    Vec score; // temporary score

    const Scalar kappa_init;
};

template <typename Derived, typename T>
void
train_marker_genes(annot_model_t &annot,
                   const Eigen::MatrixBase<Derived> &_X, // expression matrix
                   const T &options)
{

    const Mat &L = annot.L;
    const Derived &X = _X.derived();
    const Index M = L.rows();
    const Index K = L.cols();

    ASSERT(M > 1, "Must have more than two rows");

    using DS = discrete_sampler_t<Scalar, Index>;
    DS sampler_k(K); // sample discrete from log-mass

    std::random_device rd;
    std::mt19937 rgen(rd());
    std::normal_distribution<Scalar> rnorm(0., 1.);

    Mat xx = X;
    Mat mu = L;

    normalize_columns(mu);
    normalize_columns(xx);

    Scalar score;

    Vec xj(M);
    Vec nsize(K);
    Vec mass(K);
    std::vector<Index> membership(xx.cols());
    std::fill(membership.begin(), membership.end(), 0);

    const Index max_em_iter = options.max_em_iter;
    Vec score_trace(max_em_iter);

    progress_bar_t<Index> prog(max_em_iter, 1);

    ///////////////////////////////
    // Initial greedy assignment //
    ///////////////////////////////

    const Scalar pseudo = 1.0;

    Mat Stat(M, K); // feature x label
    Stat.setConstant(pseudo); // sum x(g,j) * z(j, k)

    Mat scoreMat = mu.transpose() * xx;
    nsize.setConstant(pseudo);
    const Scalar eps = 1e-2;

    for (Index j = 0; j < scoreMat.cols(); ++j) {
        xj = xx.col(j);
        Index argmax;
        const Vec &_score = annot.log_score(xj);
        _score.maxCoeff(&argmax);

        if (!options.unconstrained_update) {
            xj = xj.cwiseProduct(L.col(argmax));
        }

        membership[j] = argmax;
        nsize(argmax) += 1.0;
        Stat.col(argmax) += xj;
    }

    annot.update_param(Stat, nsize);

    if (options.verbose) {
        std::cerr << "N:     " << nsize.transpose() << "\n" << std::endl;
        std::cerr << "kappa: " << annot.kappa.transpose() << "\n" << std::endl;
    }

    ////////////////////////////
    // Memoized online update //
    ////////////////////////////

    std::vector<Index> indexes(scoreMat.cols());
    std::iota(indexes.begin(), indexes.end(), 0);

    Vec sj(K); // type x 1 score vector

    for (Index iter = 0; iter < max_em_iter; ++iter) {
        score = 0;
        std::shuffle(indexes.begin(), indexes.end(), rgen);

        for (Index j : indexes) {
            xj = xx.col(j);

            const Index k_prev = membership.at(j);

            // Index k;
            // score += sj.maxCoeff(&k);

            sj = annot.log_score(xj);
            const Index k_now = sampler_k(sj);
            score += sj(k_now);

            if (k_now != k_prev) {

                nsize(k_prev) -= 1.0;
                nsize(k_now) += 1.0;

                if (options.unconstrained_update) {
                    Stat.col(k_prev) -= xj;
                    Stat.col(k_now) += xj;
                } else {
                    Stat.col(k_prev) -= xj.cwiseProduct(L.col(k_prev));
                    Stat.col(k_now) += xj.cwiseProduct(L.col(k_now));
                }

                membership[j] = k_now;
            }
        }

        annot.update_param(Stat, nsize);

        score = score / static_cast<Scalar>(indexes.size());
        score_trace(iter) = score;

        Scalar diff = std::abs(score_trace(iter - 1) - score_trace(iter));

        prog.update();

        if (!options.verbose) {
            prog(std::cerr);
        } else {
            TLOG("[" << iter << "] [" << score << "]");
            std::cerr << "N:     " << nsize.transpose() << "\n" << std::endl;
            std::cerr << "kappa: " << annot.kappa.transpose() << "\n"
                      << std::endl;
        }

        if (iter >= 10 && diff < options.em_tol) {
            if (!options.verbose)
                std::cerr << "\r" << std::endl;
            TLOG("I[ " << iter << " ] D[ " << diff << " ]"
                       << " --> converged < " << options.em_tol);
            break;
        }
    }

    if (!options.verbose)
        std::cerr << "\r" << std::endl;
}

/////////////////////////////
// read matched annotation //
/////////////////////////////

template <typename T>
std::tuple<SpMat, std::vector<std::string>, std::vector<std::string>>
read_annotation_matched(const T &options)
{
    using Str = std::string;

    std::vector<std::tuple<Str, Str>> pair_vec;
    read_pair_file<Str, Str>(options.ann, pair_vec);

    std::vector<Str> row_vec;
    read_vector_file(options.row, row_vec);

    std::unordered_map<Str, Index> row_pos; // row name -> row index
    for (Index j = 0; j < row_vec.size(); ++j) {
        row_pos[row_vec.at(j)] = j;
    }

    std::unordered_map<Str, Index> label_pos; // label name -> label index
    {
        Index j = 0;
        for (auto pp : pair_vec) {
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

    for (auto pp : pair_vec) {
        if (row_pos.count(std::get<0>(pp)) > 0 &&
            label_pos.count(std::get<1>(pp)) > 0) {
            Index r = row_pos.at(std::get<0>(pp));
            Index l = label_pos.at(std::get<1>(pp));
            triples.push_back(ET(r, l, 1.0));
        }
    }

    SpMat L(row_pos.size(), label_pos.size());
    L.reserve(triples.size());
    L.setFromTriplets(triples.begin(), triples.end());

    std::vector<Str> labels(label_pos.size());
    std::vector<Str> rows(row_pos.size());

    for (auto pp : label_pos) {
        labels[std::get<1>(pp)] = std::get<0>(pp);
    }

    for (auto pp : row_pos) {
        rows[std::get<1>(pp)] = std::get<0>(pp);
    }

    if (options.verbose)
        for (auto l : labels) {
            TLOG("Annotation Labels: " << l);
        }

    return std::make_tuple(L, rows, labels);
}

//////////////////////
// Argument parsing //
//////////////////////

struct annotate_options_t {
    using Str = std::string;

    typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
    const std::vector<Str> METHOD_NAMES;

    annotate_options_t()
    {
        mtx = "";
        col = "";
        row = "";
        ann = "";
        out = "output.txt.gz";

        col_norm = 10000;

        raw_scale = true;
        log_scale = false;

        initial_sample = 100000;
        batch_size = 100000;

        max_em_iter = 100;

        sampling_method = CV;

        em_tol = 1e-4;

        tau = 1.0;
        rank = 50;
        lu_iter = 5;

        balance_marker_size = false;
        unconstrained_update = false;
        verbose = false;
    }

    Str mtx;
    Str col;
    Str row;
    Str ann;
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

    Scalar tau;
    Index rank;
    Index lu_iter;

    bool balance_marker_size;
    bool unconstrained_update;
    bool verbose;
};

template <typename T>
int
parse_annotate_options(const int argc, //
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
        "--ann (-a)             : row annotation file\n"
        "--out (-o)             : Output file header\n"
        "\n"
        "--unconstrained (-d)   : Unconstrained dictionary (default: false)\n"
        "--log_scale (-L)       : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)       : Data in a raw-scale (default: true)\n"
        "\n"
        "--initial_sample (-I)  : Initial sample size (default: 100000)\n"
        "--batch_size (-B)      : Batch size (default: 100000)\n"
        "--sampling_method (-M) : Sampling method: CV (default), "
        "MEAN, UNIFORM\n"
        "\n"
        "--em_iter (-i)         : EM iteration (default: 100)\n"
        "--em_tol (-e)          : EM convergence criterion (default: 1e-4)\n"
        "\n"
        "--tau (-u)             : Regularization parameter (default: tau = 1)\n"
        "--rank (-r)            : The maximal rank of SVD (default: rank = 50)\n"
        "--iter (-l)            : # of LU iterations (default: iter = 5)\n"
        "\n"
        "--verbose (-v)         : Set verbose (default: false)\n"
        "--balance_markers (-k) : Balance maker size (default: false)\n"
        "\n";

    const char *const short_opts = "m:c:f:a:o:I:B:M:LRi:e:hkd:u:r:l:v";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' }, //
          { "data", required_argument, nullptr, 'm' }, //
          { "col", required_argument, nullptr, 'c' }, //
          { "row", required_argument, nullptr, 'f' }, //
          { "feature", required_argument, nullptr, 'f' }, //
          { "ann", required_argument, nullptr, 'a' }, //
          { "out", required_argument, nullptr, 'o' }, //
          { "log_scale", no_argument, nullptr, 'L' }, //
          { "raw_scale", no_argument, nullptr, 'R' }, //
          { "initial_sample", required_argument, nullptr, 'I' }, //
          { "batch_size", required_argument, nullptr, 'B' }, //
          { "sampling_method", required_argument, nullptr, 'M' }, //
          { "em_iter", required_argument, nullptr, 'i' }, //
          { "em_tol", required_argument, nullptr, 'e' }, //
          { "help", no_argument, nullptr, 'h' }, //
          { "balance_marker", no_argument, nullptr, 'k' }, //
          { "balance_markers", no_argument, nullptr, 'k' }, //
          { "unconstrained_update", no_argument, nullptr, 'd' }, //
          { "unconstrained", no_argument, nullptr, 'd' }, //
          { "tau", required_argument, nullptr, 'u' }, //
          { "rank", required_argument, nullptr, 'r' }, //
          { "lu_iter", required_argument, nullptr, 'l' }, //
          { "verbose", no_argument, nullptr, 'v' }, //
          { nullptr, no_argument, nullptr, 0 } };

    while (true) {
        const auto opt = getopt_long(argc, //
                                     const_cast<char **>(argv), //
                                     short_opts, //
                                     long_opts, //
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

        case 'o':
            options.out = std::string(optarg);
            break;

        case 'i':
            options.max_em_iter = std::stoi(optarg);
            break;

        case 'e':
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
        case 'k': // -k or --balance_marker
            options.balance_marker_size = true;
            break;
        case 'd':
            options.unconstrained_update = true;
            break;
        case 'u':
            options.tau = std::stof(optarg);
            break;
        case 'r':
            options.rank = std::stoi(optarg);
            break;
        case 'l':
            options.lu_iter = std::stoi(optarg);
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

#endif
