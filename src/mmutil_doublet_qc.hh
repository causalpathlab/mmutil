#include <getopt.h>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "eigen_util.hh"
#include "inference/sampler.hh"
#include "io.hh"
#include "mmutil.hh"
#include "mmutil_index.hh"
#include "mmutil_normalize.hh"
#include "mmutil_stat.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "utils/progress.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "ext/tabix/bgzf.h"

#include "inference/sampler.hh"
#include "link_community.hh"

#ifndef MMUTIL_DOUBLET_QC_HH_
#define MMUTIL_DOUBLET_QC_HH_

struct doublet_qc_options_t {
    using Str = std::string;

    doublet_qc_options_t()
    {
        mtx = "";
        col = "";
        row = "";
        argmax_file = "";
        lab_file = "";
        out = "output";
        row_weight_file = "";

        tau = 1.0;
        rank = 50;
        lu_iter = 5;
        col_norm = 1000;

        raw_scale = true;
        log_scale = false;

        knn = 10;
        bilink = 5; // 2 ~ 100 (bi-directional link per element)
        nlist = 51; // knn ~ N (nearest neighbour)

        batch_size = 100000;
        verbose = false;

        em_iter = 10;
        em_tol = 1e-2;
        em_recalibrate = true;

        lc_ngibbs = 500;
        lc_nlocal = 10;
        lc_nburnin = 10;

        doublet_cutoff = 2.0;
    }

    Str mtx;
    Str col;
    Str row;
    Str argmax_file;
    Str lab_file;
    Str row_weight_file;
    Str out;

    Scalar tau;
    Index rank;
    Index lu_iter;
    Scalar col_norm;

    Index knn;
    Index bilink;
    Index nlist;

    bool raw_scale;
    bool log_scale;

    Index batch_size;
    bool verbose;

    Index em_iter;
    Scalar em_tol;
    bool em_recalibrate;

    Index lc_ngibbs;
    Index lc_nlocal;
    Index lc_nburnin;

    Scalar doublet_cutoff;
};

template <typename T>
int
parse_doublet_qc_options(const int argc,     //
                         const char *argv[], //
                         T &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)             : data MTX file\n"
        "--data (-m)            : data MTX file\n"
        "--feature (-f)         : data row file (features)\n"
        "--row (-f)             : data row file (features)\n"
        "--assign (-x)          : membership assignment\n"
        "--argmax (-x)          : membership assignment\n"
        "--label (-q)           : label name file\n"
        // "\n"
        // "--dbl (-d)             : doublet t-stat cutoff (default: 2)\n"
        "\n"
        "--knn (-k)             : k nearest neighbors (default: 50)\n"
        "--bilink (-b)          : # of bidirectional links (default: 5)\n"
        "--nlist (-n)           : # nearest neighbor lists (default: 51)\n"
        "--out (-o)             : Output file header\n"
        "\n"
        "--rank (-r)            : # of SVD factors (default: rank = 50)\n"
        "--iter (-l)            : # of LU iterations (default: iter = 5)\n"
        "--row_weight (-w)      : Feature re-weighting (default: none)\n"
        "--col_norm (-C)        : Column normalization (default: 10000)\n"
        "\n"
        "--log_scale (-L)       : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)       : Data in a raw-scale (default: true)\n"
        "\n"
        "--gibbs (-G)           : # Gibbs sampling (default: 500)\n"
        "--local (-A)           : # Block-Gibbs (local) sampling (default: 10)\n"
        "--burnin (-U)          : # Burn-in sampling (default: 10)\n"
        "\n"
        "--verbose (-v)         : Set verbose (default: false)\n"
        "\n"
        "[Details for kNN graph]\n"
        "\n"
        "(M)\n"
        "The number of bi-directional links created for every new element  \n"
        "during construction. Reasonable range for M is 2-100. Higher M work \n"
        "better on datasets with intrinsic dimensionality and/or high recall, \n"
        "while low M works better for datasets intrinsic dimensionality and/or\n"
        "low recalls. \n"
        "\n"
        "(N)\n"
        "The size of the dynamic list for the nearest neighbors (used during \n"
        "the search). A higher more accurate but slower search. This cannot be\n"
        "set lower than the number nearest neighbors k. The value ef of can be \n"
        "anything between of the dataset. [Reference] Malkov, Yu, and Yashunin. "
        "\n"
        "`Efficient and robust approximate nearest neighbor search using \n"
        "Hierarchical Navigable Small World graphs.` \n"
        "\n"
        "preprint: "
        "https://arxiv.org/abs/1603.09320 See also: "
        "https://github.com/nmslib/hnswlib"
        "\n";

    const char *const short_opts = "m:x:f:q:r:l:C:w:d:k:b:n:o:B:LRi:t:G:A:U:hv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "col", required_argument, nullptr, 'c' },        //
          { "row", required_argument, nullptr, 'f' },        //
          { "feature", required_argument, nullptr, 'f' },    //
          { "argmax", required_argument, nullptr, 'x' },     //
          { "assignment", required_argument, nullptr, 'x' }, //
          { "assign", required_argument, nullptr, 'x' },     //
          { "lab", required_argument, nullptr, 'q' },        //
          { "label", required_argument, nullptr, 'q' },      //
          { "rank", required_argument, nullptr, 'r' },       //
          { "lu_iter", required_argument, nullptr, 'l' },    //
          { "row_weight", required_argument, nullptr, 'w' }, //
          { "col_norm", required_argument, nullptr, 'C' },   //
          { "dbl", required_argument, nullptr, 'd' },        //
          { "doublet", required_argument, nullptr, 'd' },    //
          { "knn", required_argument, nullptr, 'k' },        //
          { "bilink", required_argument, nullptr, 'b' },     //
          { "nlist", required_argument, nullptr, 'n' },      //
          { "out", required_argument, nullptr, 'o' },        //
          { "log_scale", no_argument, nullptr, 'L' },        //
          { "raw_scale", no_argument, nullptr, 'R' },        //
          { "batch_size", required_argument, nullptr, 'B' }, //
          { "em_iter", required_argument, nullptr, 'i' },    //
          { "em_tol", required_argument, nullptr, 't' },     //
          { "gibbs", required_argument, nullptr, 'G' },      //
          { "ngibbs", required_argument, nullptr, 'G' },     //
          { "local", required_argument, nullptr, 'A' },      //
          { "nlocal", required_argument, nullptr, 'A' },     //
          { "burnin", required_argument, nullptr, 'U' },     //
          { "nburnin", required_argument, nullptr, 'U' },    //
          { "help", no_argument, nullptr, 'h' },             //
          { "verbose", no_argument, nullptr, 'v' },          //
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

        case 'C':
            options.col_norm = std::stof(optarg);
            break;
        case 'r':
            options.rank = std::stoi(optarg);
            break;
        case 'l':
            options.lu_iter = std::stoi(optarg);
            break;
        case 'w':
            options.row_weight_file = std::string(optarg);
            break;

        case 'o':
            options.out = std::string(optarg);
            break;

        case 'x':
            options.argmax_file = std::string(optarg);
            break;

        case 'q':
            options.lab_file = std::string(optarg);
            break;

        case 'd':
            options.doublet_cutoff = std::stoi(optarg);
            break;

        case 'k':
            options.knn = std::stoi(optarg);
            break;

        case 'b':
            options.bilink = std::stoi(optarg);
            break;
        case 'n':
            options.nlist = std::stoi(optarg);
            break;

        case 'i':
            options.em_iter = std::stoi(optarg);
            break;

        case 't':
            options.em_tol = std::stof(optarg);
            break;

        case 'B':
            options.batch_size = std::stoi(optarg);
            break;

        case 'G':
            options.lc_ngibbs = std::stoi(optarg);
            break;

        case 'A':
            options.lc_nlocal = std::stoi(optarg);
            break;

        case 'U':
            options.lc_nburnin = std::stoi(optarg);
            break;

        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;
        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;
        case 'v': // -v or --verbose
            options.verbose = true;
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
    ERR_RET(!file_exists(options.argmax_file), "No argmax file");
    ERR_RET(!file_exists(options.lab_file), "No label file");
    ERR_RET(options.rank < 2, "Too small rank");

    return EXIT_SUCCESS;
}

/**
 * @param deg_i number of elements
 * @param dist deg_i-vector for distance
 * @param weights deg_i-vector for weights

 Since the inner-product distance is d(x,y) = (1 - x'y),
 d = 0.5 * (x - y)'(x - y) = 0.5 * (x'x + y'y) - x'y,
 we have Gaussian weight w(x,y) = exp(-lambda * d(x,y))

 */
inline void
normalize_weights(const Index deg_i,
                  std::vector<Scalar> &dist,
                  std::vector<Scalar> &weights)
{
    ASSERT(deg_i > 0, "mmutil_doublet_qc: deg_i > 0");
    ASSERT(dist.size() >= deg_i, "mmutil_doublet_qc: At least deg_i");
    ASSERT(dist.size() == weights.size(),
           "mmutil_doublet_qc: check distance and weights");

    const Scalar _log2 = fasterlog(2.);
    const Scalar _di = static_cast<Scalar>(deg_i);
    const Scalar log2K = fasterlog(_di) / _log2;

    Scalar lambda = 1.0;

    const Scalar dmin = *std::min_element(dist.begin(), dist.begin() + deg_i);

    auto f = [&](const Scalar lam) -> Scalar {
        Scalar rhs = 0.;
        for (Index j = 0; j < deg_i; ++j) {
            Scalar w = fasterexp(-(dist[j] - dmin) * lam);
            rhs += w;
        }
        Scalar lhs = log2K * lam;
        return (lhs - rhs);
    };

    Scalar fval = f(lambda);

    while (true) {
        Scalar _lam = lambda;
        if (fval < 0.) {
            _lam = lambda * 1.1;
        } else {
            _lam = lambda * 0.9;
        }
        Scalar _fval = f(_lam);
        if (std::abs(_fval) > std::abs(fval)) {
            break;
        }
        lambda = _lam;
        fval = _fval;
    }

    for (Index j = 0; j < deg_i; ++j) {
        weights[j] = fasterexp(-(dist[j] - dmin) * lambda);
    }
}

/**
 * @param options
 */
struct collector_t {

    struct NS : public check_positive_t<Index> {
        explicit NS(const Index n)
            : check_positive_t<Index>(n)
        {
        }
    };

    struct ND : public check_positive_t<Index> {
        explicit ND(const Index n)
            : check_positive_t<Index>(n)
        {
        }
    };

    struct KK : public check_positive_t<Index> {
        explicit KK(const Index n)
            : check_positive_t<Index>(n)
        {
        }
    };

    explicit collector_t(const NS ns, const ND nd, const KK kk)
        : n_singlet(ns.val)
        , n_doublet(nd.val)
        , K(kk.val)
        , DegS(K, 1)
        , DegD(K, 1)
        , TotS(K, 1)
        , TotD(K, 1)
        , DegNK(n_singlet, K)
        , PropNK(n_singlet + n_doublet, K)
        , denomD(
              Mat::Constant(n_doublet, 1, 1. / static_cast<Scalar>(n_doublet)))
        , denomS(
              Mat::Constant(n_singlet, 1, 1. / static_cast<Scalar>(n_singlet)))
    {
    }

    inline void add(lc_model_t &_lc)
    {
        DegS(_lc.Deg.leftCols(n_singlet) * denomS);
        DegD(_lc.Deg.rightCols(n_doublet) * denomD);
        TotS(_lc.Prop.leftCols(n_singlet) * denomS);
        TotD(_lc.Prop.rightCols(n_doublet) * denomD);
        DegNK(_lc.Deg.leftCols(n_singlet).transpose());
        PropNK(_lc.Prop.transpose());
    }

    inline void operator()(lc_model_t &_lc) { add(_lc); }

    void write_singlet_degree(std::string file_name)
    {

        Mat deg_mu = DegNK.mean();
        Mat deg_sd = DegNK.var().cwiseSqrt();

        using _tup = std::tuple<Index, Index, Scalar, Scalar>;
        std::vector<_tup> ret;
        ret.reserve(n_singlet * K);

        for (Index k = 0; k < K; ++k) {
            for (Index i = 0; i < n_singlet; ++i) {
                ret.emplace_back(i, k, deg_mu(i, k), deg_sd(i, k));
            }
        }
        write_tuple_file(file_name, ret);
    }

    void write_prop_stat(std::string mean_name, std::string sd_name)
    {
        write_data_file(mean_name, PropNK.mean());
        write_data_file(sd_name, PropNK.var().cwiseSqrt());
    }

    void write_cluster_stat(std::string file_name)
    {
        Vec sing_deg_mu = DegS.mean();
        Vec sing_deg_sd = DegS.var().cwiseSqrt();
        Vec dbl_deg_mu = DegD.mean();
        Vec dbl_deg_sd = DegD.var().cwiseSqrt();

        Vec sing_tot_mu = TotS.mean();
        Vec sing_tot_sd = TotS.var().cwiseSqrt();
        Vec dbl_tot_mu = TotD.mean();
        Vec dbl_tot_sd = TotD.var().cwiseSqrt();

        using _tup = std::tuple<Index,
                                Scalar,
                                Scalar,
                                Scalar,
                                Scalar,
                                Scalar,
                                Scalar,
                                Scalar,
                                Scalar>;

        std::vector<_tup> ret;
        ret.reserve(K);

        for (Index k = 0; k < K; ++k) {
            ret.emplace_back(k,
                             sing_deg_mu(k),
                             sing_deg_sd(k),
                             dbl_deg_mu(k),
                             dbl_deg_sd(k),
                             sing_tot_mu(k),
                             sing_tot_sd(k),
                             dbl_tot_mu(k),
                             dbl_tot_sd(k));
        }

        write_tuple_file(file_name, ret);
    }

    void write_doublet_score(const std::string file_name,
                             const std::vector<std::string> &_names)
    {
        ASSERT(_names.size() >= n_singlet,
               "mmutil_doublet_qc: Insufficient names");

        using _tup = std::tuple<Index, std::string, Scalar, Scalar, Scalar>;
        std::vector<_tup> ret;
        ret.reserve(n_singlet);

        Vec deg_mu = DegNK.mean().col(0);
        Vec deg_sd = DegNK.var().col(0).cwiseSqrt();
        Vec _mu0 = DegD.mean();
        Vec _sd0 = DegD.var().cwiseSqrt();
        Scalar mu0 = _mu0(0), sd0 = _sd0(0);

        const Scalar reg = 1e-2;

        for (Index i = 0; i < n_singlet; ++i) {
            const Scalar denom = sd0 * sd0 + deg_sd(i) * deg_sd(i);
            const Scalar z = (deg_mu(i) - mu0) / std::sqrt(denom + reg);
            ret.emplace_back(i, _names[i], deg_mu(i), deg_sd(i), z);
        }

        write_tuple_file(file_name, ret);
    }

    const Index n_singlet;
    const Index n_doublet;
    const Index K;

    running_stat_t<Vec> DegS;
    running_stat_t<Vec> DegD;
    running_stat_t<Vec> TotS;
    running_stat_t<Vec> TotD;
    running_stat_t<Mat> DegNK;
    running_stat_t<Mat> PropNK;

    const Mat denomD;
    const Mat denomS;
};

/**
 * @param options
 */
int
run_doublet_qc(doublet_qc_options_t &options)
{

    const std::string mtx_file = options.mtx;
    const std::string idx_file = mtx_file + ".index";

    std::vector<std::string> columns;
    CHK_ERR_RET(read_vector_file(options.col, columns),
                "Failed to read the column file: " << options.col);

    CHK_ERR_RET(mmutil::bgzf::convert_bgzip(mtx_file),
                "Failed to obtain a bgzipped file: " << mtx_file);

    CHK_ERR_RET(mmutil::index::build_mmutil_index(mtx_file, idx_file),
                "Failed to construct an index file: " << idx_file);

    std::vector<mmutil::index::idx_pair_t> idx_tab;
    CHECK(mmutil::index::read_mmutil_index(idx_file, idx_tab));

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index N = info.max_col;

    TLOG("N = " << N);
    ASSERT_RET(columns.size() == N, "Must have N x 1 column names");

    std::unordered_map<std::string, Index> column_position;

    for (Index j = 0; j < N; ++j)
        column_position[columns[j]] = j;

    std::vector<std::string> labels;
    CHK_ERR_RET(read_vector_file(options.lab_file, labels),
                "Failed to read the label file: " << options.lab_file);

    std::unordered_map<std::string, Index> label_position;

    const Index K = labels.size();
    for (Index k = 0; k < K; ++k)
        label_position[labels[k]] = k;

    ////////////////////////////////////
    // cell-level assignment results  //
    ////////////////////////////////////
    std::vector<std::tuple<std::string, std::string>> argmax_vec;
    argmax_vec.reserve(N);
    CHECK(read_pair_file(options.argmax_file, argmax_vec));
    ASSERT(argmax_vec.size() >= N, "Need membership for each column");

    std::vector<Index> membership(N);
    std::fill(membership.begin(), membership.end(), -1);

    for (auto pp : argmax_vec) {
        if (column_position.count(std::get<0>(pp)) > 0 &&
            label_position.count(std::get<1>(pp)) > 0) {
            const Index i = column_position[std::get<0>(pp)];
            const Index k = label_position[std::get<1>(pp)];
            membership[i] = k;
        }
    }

    TLOG("Constructed the membership vector");

    std::vector<std::shared_ptr<std::vector<Index>>> index_sets;
    for (Index k = 0; k < K; ++k) {
        index_sets.emplace_back(std::make_shared<std::vector<Index>>());
    }

    for (Index j = 0; j < membership.size(); ++j) {
        const Index k = membership[j];
        if (k >= 0) {
            index_sets[k]->emplace_back(j);
        }
    }

    TLOG("Read membership");

    ///////////////////////////
    // weights for the rows  //
    ///////////////////////////

    Vec weights;
    if (file_exists(options.row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(options.row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    auto take_batch_data_subcol = [&](std::vector<Index> &subcol) -> Mat {
        using namespace mmutil::index;
        SpMat x = read_eigen_sparse_subset_col(mtx_file, idx_tab, subcol);
        return Mat(x);
    };

    auto take_batch_data = [&](Index lb, Index ub) -> Mat {
        std::vector<Index> subcol(ub - lb);
        std::iota(subcol.begin(), subcol.end(), lb);
        return take_batch_data_subcol(subcol);
    };

    const Index batch_size = options.batch_size;

    ////////////////////////////////
    // Learn latent embedding ... //
    ////////////////////////////////

    TLOG("Training SVD for spectral matching ...");
    svd_out_t svd = take_svd_online_em(mtx_file, ww, options);
    const Index rank = svd.U.cols();

    TLOG("SVD rank = " << rank);
    Mat proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
    TLOG("Found projection: " << proj.rows() << " x " << proj.cols());

    ////////////////////
    // kNN parameters //
    ////////////////////

    std::size_t knn = options.knn;
    std::size_t param_bilink = options.bilink;
    std::size_t param_nnlist = options.nlist;

    if (param_bilink >= rank) {
        WLOG("Shrink M value: " << param_bilink << " vs. " << rank);
        param_bilink = rank - 1;
    }

    if (param_bilink < 2) {
        WLOG("too small M value");
        param_bilink = 2;
    }

    if (param_nnlist <= knn) {
        WLOG("too small N value");
        param_nnlist = knn + 1;
    }

    /** Take singlet data for a particular type "k"
     * @param k type index
     */
    auto build_spectral_singlet = [&](const Index k) -> Mat {
        std::vector<Index> &col_k = *index_sets[k].get();
        const Index Nk = col_k.size();
#ifdef DEBUG
        TLOG("Nk: " << Nk);
#endif
        Mat ret(rank, Nk);
        ret.setZero();

        Index r = 0;
        for (Index lb = 0; lb < Nk; lb += batch_size) {
            const Index ub = std::min(Nk, batch_size + lb);

            std::vector<Index> subcol_k(ub - lb);
#ifdef DEBUG
            TLOG("[" << lb << ", " << ub << ")");
#endif
            std::copy(col_k.begin() + lb, col_k.begin() + ub, subcol_k.begin());

            Mat x0 = take_batch_data_subcol(subcol_k);

#ifdef DEBUG
            ASSERT(x0.cols() == subcol_k.size(), "singlet: size doesn't match");
#endif

            Mat xx = make_normalized_laplacian(x0,
                                               ww,
                                               options.tau,
                                               options.col_norm,
                                               options.log_scale);

#ifdef DEBUG
            TLOG("X: " << xx.rows() << " x " << xx.cols());
#endif
            Mat vv = proj.transpose() * xx; // rank x batch_size
            normalize_columns(vv);
#ifdef DEBUG
            TLOG("V: " << vv.rows() << " x " << vv.cols());
#endif
            for (Index j = 0; j < vv.cols(); ++j) {
                ret.col(r) = vv.col(j);
                ++r;
            }
        }

        TLOG("Found singlet matrix: " << ret.rows() << " x " << ret.cols());

        return ret;
    };

    /** Take doublet data for a particular type "k"
     *	@param k type index
     */
    auto build_spectral_doublet = [&](const Index k) -> Mat {
        Index Nrnd = 0;
        std::random_device rd;
        std::mt19937 rg(rd());

        std::vector<Index> &_col_k = *index_sets[k].get();

        std::vector<Index> col_k(_col_k.size());
        std::copy(_col_k.begin(), _col_k.end(), col_k.begin());
        std::shuffle(col_k.begin(), col_k.end(), rg);

        for (Index l = 0; l < K; ++l) {
            if (l == k)
                continue;
            std::vector<Index> &col_l = *index_sets[l].get();
            const Index nkl = std::min(col_k.size(), col_l.size());
            if (nkl < 1)
                continue;

            Nrnd += nkl;
        }
#ifdef DEBUG
        TLOG("Nrnd: " << Nrnd);
#endif
        Mat ret(rank, Nrnd);
        ret.setZero();

        Index r = 0;
        for (Index l = 0; l < K; ++l) {
            if (l == k)
                continue;

            std::vector<Index> &_col_l = *index_sets[l].get();

            const Index nkl = std::min(col_k.size(), _col_l.size());
            if (nkl < 1)
                continue;

            std::vector<Index> col_l(_col_l.size());
            std::copy(_col_l.begin(), _col_l.end(), col_l.begin());
            std::shuffle(col_l.begin(), col_l.end(), rg);
#ifdef DEBUG
            TLOG("Shuffled doublet indexes: Mixing clusters "
                 << k << " with " << l << " -> " << nkl << " elements");
#endif
            for (Index lb = 0; lb < nkl; lb += batch_size) {
                const Index ub = std::min(nkl, batch_size + lb);
#ifdef DEBUG
                TLOG("[" << lb << ", " << ub << ")");
#endif
                std::vector<Index> subcol_k(ub - lb);
                std::vector<Index> subcol_l(ub - lb);

                std::copy(col_k.begin() + lb,
                          col_k.begin() + ub,
                          subcol_k.begin());

                std::copy(col_l.begin() + lb,
                          col_l.begin() + ub,
                          subcol_l.begin());

                Mat xx = take_batch_data_subcol(subcol_k);
#ifdef DEBUG
                ASSERT(xx.cols() == subcol_k.size(),
                       "doublet k: size doesn't match");
#endif
                Mat yy = take_batch_data_subcol(subcol_l);
#ifdef DEBUG
                ASSERT(yy.cols() == subcol_l.size(),
                       "doublet l: size doesn't match");
#endif
                Mat xy0 = xx * .5 + yy * .5;

                Mat xy = make_normalized_laplacian(xy0,
                                                   ww,
                                                   options.tau,
                                                   options.col_norm,
                                                   options.log_scale);

                Mat vv = proj.transpose() * xy; // rank x batch_size
                normalize_columns(vv);
                for (Index j = 0; j < vv.cols(); ++j) {
                    ret.col(r) = vv.col(j);
                    ++r;
                }
            }
#ifdef DEBUG
            TLOG("Built doublet matrix: " << k << " with " << l);
#endif
        }

        TLOG("Found doublet matrix: " << ret.rows() << " x " << ret.cols());
        return ret;
    };

    /////////////////////////////////
    // doublet Q/C for each type k //
    /////////////////////////////////

    for (Index k = 0; k < K; ++k) {

        std::vector<Index> &col_k = *index_sets[k].get();
        if (col_k.size() < 10)
            continue;

        std::string lab_k = labels[k];
        TLOG("Checking doublets on " << lab_k);

        Mat ss = build_spectral_singlet(k);
        Mat dd = build_spectral_doublet(k);

        const Index n_singlet = ss.cols();
        const Index n_doublet = dd.cols();
        const Index n_tot = n_singlet + n_doublet;

        TLOG(n_singlet << " singlets vs. " << n_doublet << " doublets");

        // std::vector<std::tuple<Index, Index, Scalar>> knn_index;
        // const Index nquery = (n_singlet) > knn ? knn : (n_singlet - 1);

        // // singlet-singlet edges
        // {
        //     index_triplet_vec _index;
        //     CHECK(search_knn(SrcDataT(ss.data(), rank, n_singlet),
        //                      TgtDataT(ss.data(), rank, n_singlet),
        //                      KNN(nquery),
        //                      BILINK(param_bilink),
        //                      NNLIST(param_nnlist),
        //                      _index));

        //     Index i, j;
        //     Scalar d;
        //     for (auto tt : _index) {
        //         std::tie(i, j, d) = tt;
        //         if (i == j)
        //             continue;
        //         knn_index.emplace_back(i, j, 1. - d);
        //     }
        // }

        // // singlet-doublet edges
        // {
        //     index_triplet_vec _index;
        //     CHECK(search_knn(SrcDataT(ss.data(), rank, n_singlet),
        //                      TgtDataT(dd.data(), rank, n_doublet),
        //                      KNN(nquery),
        //                      BILINK(param_bilink),
        //                      NNLIST(param_nnlist),
        //                      _index));
        //     Index i, _j;
        //     Scalar d;
        //     for (auto tt : _index) {
        //         std::tie(i, _j, d) = tt;
        //         const Index j = _j + n_singlet;
        //         knn_index.emplace_back(i, j, 1. - d);
        //     }
        // }

        // // doublet-doublet edges
        // {
        //     index_triplet_vec _index;
        //     CHECK(search_knn(SrcDataT(dd.data(), rank, n_doublet),
        //                      TgtDataT(dd.data(), rank, n_doublet),
        //                      KNN(nquery),
        //                      BILINK(param_bilink),
        //                      NNLIST(param_nnlist),
        //                      _index));
        //     Index _i, _j;
        //     Scalar d;
        //     for (auto tt : _index) {
        //         std::tie(_i, _j, d) = tt;
        //         const Index i = _i + n_singlet;
        //         const Index j = _j + n_singlet;
        //         if (i == j)
        //             continue;
        //         knn_index.emplace_back(i, j, 1. - d);
        //     }
        // }

        std::vector<std::tuple<Index, Index, Scalar>> knn_index;
        const Index nquery = (n_singlet) > knn ? knn : (n_singlet - 1);

        hnswlib::InnerProductSpace VS(rank);
        KnnAlg alg(&VS, n_tot, param_bilink, param_nnlist);

        TLOG("Adding singlet points...");
        {
            float *mass = ss.data(); // adding singlet points
            progress_bar_t<Index> prog(n_singlet, 1e2);
            for (Index i = 0; i < n_singlet; ++i) {
                alg.addPoint((void *)(mass + rank * i), i);
                prog.update();
                prog(std::cerr);
            }
        }

        TLOG("Adding doublet points...");
        {
            float *mass = dd.data(); // adding doublet points
            progress_bar_t<Index> prog(n_doublet, 1e2);
            for (Index i = n_singlet; i < n_tot; ++i) {
                std::size_t j = (i - n_singlet);
                alg.addPoint((void *)(mass + rank * j), i);
                prog.update();
                prog(std::cerr);
            }
        }
        TLOG("Added " << n_tot << " points");

        std::vector<Scalar> dist(K);
        std::vector<Scalar> weights(K);
        std::vector<Index> neigh(K);
        {
            float *mass = ss.data();
            for (Index ii = 0; ii < n_singlet; ++ii) {
                std::size_t jj = ii;
                auto pq = alg.searchKnn((void *)(mass + rank * jj), nquery);
                Index deg_i = 0;
                while (!pq.empty()) {
                    float d = 0;
                    std::size_t k;
                    std::tie(d, k) = pq.top();
                    if (k != ii) {
                        if (deg_i < dist.size()) {
                            dist[deg_i] = d;
                            neigh[deg_i] = k;
                            ++deg_i;
                        }
                    }
                    pq.pop();
                }
                if (deg_i < 1)
                    continue;
                normalize_weights(deg_i, dist, weights);
                for (Index j = 0; j < deg_i; ++j) {
                    const Index k = neigh[j];
                    const Scalar w = weights[j];
                    ASSERT(w > .0, "must be non-negative");
                    knn_index.emplace_back(ii, k, w);
                }
            }
        }
        {
            float *mass = dd.data();
            for (Index ii = n_singlet; ii < n_tot; ++ii) {
                std::size_t jj = (ii - n_singlet);
                auto pq = alg.searchKnn((void *)(mass + rank * jj), nquery);
                Index deg_i = 0;
                while (!pq.empty()) {
                    float d = 0;
                    std::size_t k;
                    std::tie(d, k) = pq.top();
                    if (k != ii) {
                        if (deg_i < dist.size()) {
                            dist[deg_i] = d;
                            neigh[deg_i] = k;
                            ++deg_i;
                        }
                    }
                    pq.pop();
                }
                if (deg_i < 1)
                    continue;
                normalize_weights(deg_i, dist, weights);
                for (Index j = 0; j < deg_i; ++j) {
                    const Index k = neigh[j];
                    const Scalar w = weights[j];
                    ASSERT(w > .0, "must be non-negative");
                    knn_index.emplace_back(ii, k, w);
                }
            }
        }

        TLOG("Constructed KNN graph");

        auto lc_ptr = build_lc_model(knn_index, K);
        auto &lc = *lc_ptr.get();

        update_latent_random(lc, 1, K); // randomly assign membership

        std::unordered_set<Index> clamp;

        for (Index e = 0; e < lc.m; ++e) { // assign the membership of singlets
            Index imax = 0;
            for (SpMat::InnerIterator it(lc.Y, e); it; ++it) {
                imax = std::max(imax, it.col());
            }
            if (imax < n_singlet) {    // Both end points are singlets
                lc.Z.col(e).setZero(); //
                lc.Z(0, e) = 1.;       //
                clamp.insert(e);       // Do not change
            }
        }

        TLOG("Assigned link community membership");

        const Index nlocal = options.lc_nlocal;
        const Index ngibbs = options.lc_ngibbs;
        const Index nburnin = options.lc_nburnin;

        collector_t stat_fun{ collector_t::NS(n_singlet),
                              collector_t::ND(n_doublet),
                              collector_t::KK(K) };

        run_gibbs_sampling(lc,
                           clamp,
                           stat_fun,
                           ngibbs_t(ngibbs),
                           nburnin_t(nburnin),
                           nlocal_t(nlocal));

        std::vector<std::string> _names(n_singlet);
        for (Index j = 0; j < n_singlet; ++j) {
            _names[j] = columns[col_k[j]];
        }

        std::string outhdr = options.out + "." + lab_k;

        write_vector_file(outhdr + ".cols.gz", _names);
        stat_fun.write_cluster_stat(outhdr + ".clust.gz");
        stat_fun.write_singlet_degree(outhdr + ".deg.gz");
        stat_fun.write_prop_stat(outhdr + ".mean.gz", outhdr + ".sd.gz");

        write_tuple_file(outhdr + ".network.gz", knn_index);

        stat_fun.write_doublet_score(outhdr + ".qc.gz", _names);
    }

    return EXIT_SUCCESS;
}

#endif
