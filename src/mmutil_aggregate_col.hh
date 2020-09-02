#include <getopt.h>
#include <unordered_map>
#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"

#include "eigen_util.hh"
#include "io.hh"

#include "utils/progress.hh"
#include "inference/sampler.hh"

#include "mmutil_aggregator.hh"

#ifndef MMUTIL_AGGREGATE_COL_HH_
#define MMUTIL_AGGREGATE_COL_HH_

struct aggregate_options_t {
    using Str = std::string;

    aggregate_options_t()
    {
        mtx_file = "";
        annot_prob_file = "";
        annot_file = "";
        ind_file = "";
        lab_file = "";
        trt_ind_file = "";
        out = "output";
        verbose = false;

        tau = 1.0;
        rank = 50;
        lu_iter = 5;
        knn = 1;
        bilink = 5; // 2 ~ 100 (bi-directional link per element)
        nlist = 5;  // knn ~ N (nearest neighbour)

        raw_scale = true;
        log_scale = false;

        col_norm = 1000;
        block_size = 5000;

        em_iter = 10;
        em_tol = 1e-2;

        nburnin = 10;
        ngibbs = 100;

        discretize = true;
        normalize = false;
    }

    Str mtx_file;
    Str annot_prob_file;
    Str annot_file;
    Str col_file;
    Str ind_file;
    Str trt_ind_file;
    Str lab_file;
    Str out;

    // SVD and matching
    Str row_weight_file;

    bool raw_scale;
    bool log_scale;

    Scalar tau;
    Index rank;
    Index lu_iter;
    Scalar col_norm;

    Index knn;
    Index bilink;
    Index nlist;

    // SVD
    Index block_size;
    Index em_iter;
    Scalar em_tol;

    bool verbose;

    // aggregator
    Index nburnin;
    Index ngibbs;

    bool discretize;
    bool normalize;
};

template <typename OPTIONS>
int
parse_aggregate_options(const int argc,     //
                        const char *argv[], //
                        OPTIONS &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)        : data MTX file (M x N)\n"
        "--data (-m)       : data MTX file (M x N)\n"
        "--col (-c)        : data column file (N x 1)\n"
        "--annot (-a)      : annotation/clustering assignment (N x 2)\n"
        "--annot_prob (-A) : annotation/clustering probability (N x K)\n"
        "--ind (-i)        : N x 1 sample to individual (n)\n"
        "--trt_ind (-t)    : N x 1 sample to case-control membership\n"
        "--lab (-l)        : K x 1 annotation label name (e.g., cell type) \n"
        "--out (-o)        : Output file header\n"
        "\n"
        "[Options]\n"
        "--gibbs (-g)      : number of gibbs sampling (default: 100)\n"
        "--burnin (-G)     : number of burn-in sampling (default: 10)\n"
        "--discretize (-D) : Use discretized annotation matrix (default: true)\n"
        "--probabilistic (-P) : Use expected annotation matrix (default: false)\n"
        "\n"
        "[Counterfactual matching options]\n"
        "\n"
        "--knn (-k)        : k nearest neighbours (default: 1)\n"
        "--bilink (-b)     : # of bidirectional links (default: 5)\n"
        "--nlist (-n)      : # nearest neighbor lists (default: 5)\n"
        "\n"
        "--rank (-r)       : # of SVD factors (default: rank = 50)\n"
        "--lu_iter (-u)    : # of LU iterations (default: iter = 5)\n"
        "--row_weight (-w) : Feature re-weighting (default: none)\n"
        "--col_norm (-C)   : Column normalization (default: 10000)\n"
        "\n"
        "--log_scale (-L)  : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)  : Data in a raw-scale (default: true)\n"
        "\n"
        "[Output]\n"
        "${out}.mean.gz    : (M x n) Mean matrix\n"
        "${out}.mu.gz      : (M x n) Scaled mean matrix\n"
        "${out}.mu_sd.gz   : (M x n) SD for mu\n"
        "${out}.cols.gz    : (n x 1) Column names\n"
        "\n"
        "[Details for kNN graph]\n"
        "\n"
        "(bilink)\n"
        "The number of bi-directional links created for every new element\n"
        "during construction. Reasonable range for M is 2-100. A high M value\n"
        "works better on datasets with high intrinsic dimensionality and/or\n"
        "high recall, while a low M value works better for datasets with low\n"
        "intrinsic dimensionality and/or low recalls.\n"
        "\n"
        "(nlist)\n"
        "The size of the dynamic list for the nearest neighbors (used during\n"
        "the search). A higher N value leads to more accurate but slower\n"
        "search. This cannot be set lower than the number of queried nearest\n"
        "neighbors k. The value ef of can be anything between k and the size of\n"
        "the dataset.\n"
        "\n"
        "[Reference]\n"
        "Malkov, Yu, and Yashunin. `Efficient and robust approximate nearest\n"
        "neighbor search using Hierarchical Navigable Small World graphs.`\n"
        "\n"
        "preprint:"
        "https://arxiv.org/abs/1603.09320\n"
        "\n"
        "See also:\n"
        "https://github.com/nmslib/hnswlib\n"
        "\n";

    const char *const short_opts = "m:c:a:A:i:l:t:o:LRB:r:u:w:g:G:DPC:k:b:n:hzv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "annot_prob", required_argument, nullptr, 'A' }, //
          { "annot", required_argument, nullptr, 'a' },      //
          { "col", required_argument, nullptr, 'c' },      //
          { "ind", required_argument, nullptr, 'i' },        //
          { "trt", required_argument, nullptr, 't' },        //
          { "trt_ind", required_argument, nullptr, 't' },    //
          { "lab", required_argument, nullptr, 'l' },        //
          { "label", required_argument, nullptr, 'l' },      //
          { "out", required_argument, nullptr, 'o' },        //
          { "log_scale", no_argument, nullptr, 'L' },        //
          { "raw_scale", no_argument, nullptr, 'R' },        //
          { "block_size", required_argument, nullptr, 'B' }, //
          { "rank", required_argument, nullptr, 'r' },       //
          { "lu_iter", required_argument, nullptr, 'u' },    //
          { "row_weight", required_argument, nullptr, 'w' }, //
          { "gibbs", required_argument, nullptr, 'g' },      //
          { "burnin", required_argument, nullptr, 'G' },     //
          { "discretize", no_argument, nullptr, 'D' },       //
          { "probabilistic", no_argument, nullptr, 'P' },    //
          { "col_norm", required_argument, nullptr, 'C' },   //
          { "knn", required_argument, nullptr, 'k' },        //
          { "bilink", required_argument, nullptr, 'b' },     //
          { "nlist", required_argument, nullptr, 'n' },      //
          { "normalize", no_argument, nullptr, 'z' },        //
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
            options.mtx_file = std::string(optarg);
            break;
        case 'A':
            options.annot_prob_file = std::string(optarg);
            break;
        case 'a':
            options.annot_file = std::string(optarg);
            break;
        case 'c':
            options.col_file = std::string(optarg);
            break;
        case 'i':
            options.ind_file = std::string(optarg);
            break;
        case 't':
            options.trt_ind_file = std::string(optarg);
            break;
        case 'l':
            options.lab_file = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;
        case 'g':
            options.ngibbs = std::stoi(optarg);
            break;
        case 'G':
            options.nburnin = std::stoi(optarg);
            break;

        case 'r':
            options.rank = std::stoi(optarg);
            break;

        case 'u':
            options.lu_iter = std::stoi(optarg);
            break;

        case 'w':
            options.row_weight_file = std::string(optarg);
            break;

        case 'k':
            options.knn = std::stoi(optarg);
            break;

        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;

        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;

        case 'P':
            options.discretize = false;
            break;

        case 'D':
            options.discretize = true;
            break;

        case 'B':
            options.block_size = std::stoi(optarg);
            break;

        case 'b':
            options.bilink = std::stoi(optarg);
            break;

        case 'n':
            options.nlist = std::stoi(optarg);
            break;

        case 'z':
            options.normalize = true;
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

    ERR_RET(!file_exists(options.mtx_file), "No MTX file");
    ERR_RET(!file_exists(options.annot_prob_file) &&
                !file_exists(options.annot_file),
            "No ANNOT or ANNOT_PROB file");
    ERR_RET(!file_exists(options.ind_file), "No IND file");
    ERR_RET(!file_exists(options.lab_file), "No LAB file");

    ERR_RET(options.rank < 1, "Too small rank");

    return EXIT_SUCCESS;
}

struct cf_index_sampler_t {

    using DS = discrete_sampler_t<Scalar, Index>;

    explicit cf_index_sampler_t(const Index ntrt)
        : Ntrt(ntrt)
        , obs_idx(0)
        , cf_idx(Ntrt - 1)
        , sampler(Ntrt - 1)
        , prior_mass(Ntrt - 1)
    {
        prior_mass.setZero();
        std::iota(cf_idx.begin(), cf_idx.end(), 1);
    }

    Index operator()(const Index obs)
    {
        _resolve_cf_idx(obs);
        return cf_idx.at(sampler(prior_mass));
    }

    const Index Ntrt;

private:
    Index obs_idx;
    std::vector<Index> cf_idx;
    DS sampler;
    Vec prior_mass;

    void _resolve_cf_idx(const Index new_obs_idx)
    {
        if (new_obs_idx != obs_idx) {
            ASSERT(new_obs_idx >= 0 && new_obs_idx < Ntrt,
                   "new index must be in [0, " << Ntrt << ")");
            Index ri = 0;
            for (Index r = 0; r < Ntrt; ++r) {
                if (r != new_obs_idx)
                    cf_idx[ri++] = r;
            }
            obs_idx = new_obs_idx;
        }
    }
};

template <typename OPTIONS>
int
aggregate_col(const OPTIONS &options)
{

    const std::string mtx_file = options.mtx_file;
    const std::string idx_file = options.mtx_file + ".index";
    const std::string annot_prob_file = options.annot_prob_file;
    const std::string annot_file = options.annot_file;
    const std::string col_file = options.col_file;
    const std::string ind_file = options.ind_file;
    const std::string lab_file = options.lab_file;
    const Index ngibbs = options.ngibbs;
    const Index nburnin = options.nburnin;
    const std::string row_weight_file = options.row_weight_file;
    const std::string output = options.out;

    //////////////////
    // column names //
    //////////////////

    std::vector<std::string> cols;
    CHECK(read_vector_file(col_file, cols));
    const Index Nsample = cols.size();

    /////////////////
    // label names //
    /////////////////

    std::vector<std::string> lab_name;
    CHECK(read_vector_file(lab_file, lab_name));
    auto lab_position = make_position_dict<std::string, Index>(lab_name);
    const Index K = lab_name.size();

    ///////////////////////
    // latent annotation //
    ///////////////////////

    TLOG("Reading latent annotations");

    Mat Z;

    if (annot_file.size() > 0) {
        Z.resize(Nsample, K);
        Z.setZero();

        std::unordered_map<std::string, std::string> annot_dict;
        CHECK(read_dict_file(annot_file, annot_dict));
        for (Index j = 0; j < cols.size(); ++j) {
            const std::string &s = cols.at(j);
            if (annot_dict.count(s) > 0) {
                const std::string &t = annot_dict.at(s);
                if (lab_position.count(t) > 0) {
                    const Index k = lab_position.at(t);
                    Z(j, k) = 1.;
                }
            }
        }
    } else if (annot_prob_file.size() > 0) {
        CHECK(read_data_file(annot_prob_file, Z));
    } else {
        return EXIT_FAILURE;
    }

    ASSERT(cols.size() == Z.rows(),
           "column and annotation matrix should match");

    ASSERT(lab_name.size() == Z.cols(),
           "Need the same number of label names for the columns of Z");

    TLOG("Latent membership matrix: " << Z.rows() << " x " << Z.cols());

    ///////////////////////////
    // individual membership //
    ///////////////////////////

    std::vector<std::string> indv_membership;
    indv_membership.reserve(Z.rows());
    CHECK(read_vector_file(ind_file, indv_membership));

    ASSERT(indv_membership.size() == Z.rows(),
           "Individual membership file mismatches with Z");

    std::vector<std::string> indv_id_name;
    std::vector<Index> indv; // map: col -> indv index

    std::tie(indv, indv_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(indv_membership);

    auto indv_index_set = make_index_vec_vec(indv);

    const Index Nind = indv_id_name.size();

    TLOG("Identified " << Nind << " individuals");

    ASSERT(Z.rows() == indv.size(), "rows(Z) != rows(indv)");

    ////////////////////////////////////////////
    // case-control-like treatment membership //
    ////////////////////////////////////////////

    std::vector<std::string> trt_membership;
    trt_membership.reserve(Nsample);

    std::vector<std::string> trt_id_name;
    std::vector<Index> trt; // map: col -> trt index

    if (file_exists(options.trt_ind_file)) {

        CHECK(read_vector_file(options.trt_ind_file, trt_membership));

        ASSERT(trt_membership.size() == Nsample,
               "Treatment membership file mismatches with Z");

        std::tie(trt, trt_id_name, std::ignore) =
            make_indexed_vector<std::string, Index>(trt_membership);
    } else {
        trt.resize(Nsample);
        std::fill(trt.begin(), trt.end(), 0);
    }

    auto trt_index_set = make_index_vec_vec(trt);
    const Index Ntrt = trt_index_set.size();
    cf_index_sampler_t cf_index_sampler(Ntrt);

    TLOG("Identified " << Ntrt << " treatment conditions");

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    CHECK(mmutil::index::read_mmutil_index(idx_file, mtx_idx_tab));

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;

    ASSERT(Nsample == info.max_col, "Should have matched .mtx.gz");

    ///////////////////////////////////
    // weights for the rows/features //
    ///////////////////////////////////

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    Mat proj;

    if (Ntrt > 1) {

        ////////////////////////////////
        // Learn latent embedding ... //
        ////////////////////////////////

        TLOG("Training SVD for spectral matching ...");
        svd_out_t svd = take_svd_online_em(mtx_file, idx_file, ww, options);
        proj.resize(svd.U.rows(), svd.U.cols());
        proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
        TLOG("Found projection: " << proj.rows() << " x " << proj.cols());
    }

    /** Take a block of Y matrix
     * @param subcol
     */
    auto read_y_block = [&](std::vector<Index> &subcol) -> Mat {
        using namespace mmutil::index;
        SpMat x = read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, subcol);
        return Mat(x);
    };

    /** Take spectral data for a particular treatment group "k"
     * @param k type index
     */
    auto build_spectral_data = [&](const Index k) -> Mat {
        std::vector<Index> &col_k = trt_index_set[k];
        const Index Nk = col_k.size();
        const Index block_size = options.block_size;
        const Index rank = proj.cols();

        Mat ret(rank, Nk);
        ret.setZero();

        Index r = 0;
        for (Index lb = 0; lb < Nk; lb += block_size) {
            const Index ub = std::min(Nk, block_size + lb);

            std::vector<Index> subcol_k(ub - lb);
            // #ifdef DEBUG
            //             TLOG("[" << lb << ", " << ub << ")");
            // #endif
            std::copy(col_k.begin() + lb, col_k.begin() + ub, subcol_k.begin());

            Mat x0 = read_y_block(subcol_k);

            // #ifdef DEBUG
            //             ASSERT(x0.cols() == subcol_k.size(), "size doesn't
            //             match");
            // #endif

            Mat xx = make_normalized_laplacian(x0,
                                               ww,
                                               options.tau,
                                               options.col_norm,
                                               options.log_scale);

#ifdef DEBUG
            TLOG("X: " << xx.rows() << " x " << xx.cols());
#endif
            Mat vv = proj.transpose() * xx; // rank x block_size
            normalize_columns(vv);
#ifdef DEBUG
            TLOG("V: " << vv.rows() << " x " << vv.cols());
#endif
            for (Index j = 0; j < vv.cols(); ++j) {
                ret.col(r) = vv.col(j);
                ++r;
            }
        }
        return ret;
    };

    ////////////////////
    // kNN parameters //
    ////////////////////

    std::size_t knn = options.knn;
    std::size_t param_bilink = options.bilink;
    std::size_t param_nnlist = options.nlist;
    const Index rank = proj.cols();

    if (Ntrt > 1) {

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
    }

    ///////////////////////////////////////////////////////
    // construct dictionary for each treatment condition //
    ///////////////////////////////////////////////////////
    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_vec;
    Mat V;

    if (Ntrt > 1) {

        V.resize(rank, Nsample);

        for (Index tt = 0; tt < Ntrt; ++tt) {
            const Index n_tot = trt_index_set[tt].size();

            using vs_type = hnswlib::InnerProductSpace;

            vs_vec.push_back(std::make_shared<vs_type>(rank));

            vs_type &VS = *vs_vec[vs_vec.size() - 1].get();
            knn_lookup_vec.push_back(std::make_shared<KnnAlg>(&VS,
                                                              n_tot,
                                                              param_bilink,
                                                              param_nnlist));
        }

        progress_bar_t<Index> prog(Nsample, 1e2);

        for (Index tt = 0; tt < Ntrt; ++tt) {
            const Index n_tot = trt_index_set[tt].size();
            KnnAlg &alg = *knn_lookup_vec[tt].get();
            Mat dat = build_spectral_data(tt);
            float *mass = dat.data(); // adding data points
            for (Index i = 0; i < n_tot; ++i) {
                alg.addPoint((void *)(mass + rank * i), i);
                const Index j = trt_index_set.at(tt).at(i);
                V.col(j) = dat.col(i);
                prog.update();
                prog(std::cerr);
            }
        }
    }

    ///////////////////////////
    // For each individual i //
    ///////////////////////////

    Mat obs_mu;
    Mat obs_mu_sd;
    Mat obs_mean;
    Mat obs_sum;

    Vec obs_lambda(Nsample);
    Vec obs_lambda_sd(Nsample);

    Mat cf_mu;
    Mat cf_mu_sd;
    Mat cf_mean;
    Mat cf_sum;

    Mat wald_stat;

    Vec cf_lambda(Nsample);
    Vec cf_lambda_sd(Nsample);

    using namespace mmutil::index;

    std::vector<std::string> out_col;

    /**
     * @param i individual index [0, Nind)
     */

    auto read_cf_idx = [&](const Index i) {
        std::vector<Index> cf_indv_index;
        std::vector<Index> obs_indv_index;
        cf_indv_index.reserve(indv_index_set.at(i).size() * knn);
        obs_indv_index.reserve(indv_index_set.at(i).size() * knn);

        float *mass = V.data();

        for (Index j : indv_index_set.at(i)) {
            const Index tj = trt.at(j);
            const Index sj = cf_index_sampler(tj);
            KnnAlg &alg = *knn_lookup_vec[sj].get();

            // counter-factual data points
            const Index n_sj = trt_index_set.at(sj).size();
            const std::size_t nquery = std::min(options.knn, n_sj);

            auto pq = alg.searchKnn((void *)(mass + rank * j), nquery);
            std::size_t k;

            while (!pq.empty()) {
                std::tie(std::ignore, k) = pq.top();
                cf_indv_index.emplace_back(trt_index_set.at(sj).at(k));
                pq.pop();
            }

            // matching observed data points
            for (Index q = 0; q < nquery; ++q) {
                obs_indv_index.emplace_back(j);
            }
        }

        return std::make_tuple(cf_indv_index, obs_indv_index);
    };

    auto _sqrt = [](const Scalar &x) -> Scalar {
        return (x > 0.) ? std::sqrt(x) : 0.;
    };

    Index s_obs = 0; // cumulative for obs (be cautious; do not touch)
    Index s_cf = 0;  // cumulative for cf (be cautious; do not touch)

    const Scalar eps = 1e-4;
    const Scalar a0 = 1e-4, b0 = 1e-4;

    auto _wald_stat_fun = [&eps](const Scalar &m, const Scalar &v) -> Scalar {
        return m / std::sqrt(v + eps);
    };

    for (Index i = 0; i < Nind; ++i) {

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted at Ind = " << (i));
            return EXIT_FAILURE;
        }
#endif

        const std::string indv_name = indv_id_name.at(i);

        // Y: features x columns
        SpMat yy = read_eigen_sparse_subset_col(mtx_file,
                                                mtx_idx_tab,
                                                indv_index_set.at(i));

        if (options.normalize) {
            normalize_columns(yy);
            yy *= options.col_norm;
        }

        const Index D = yy.rows();
        const Index N = yy.cols();

        if (i == 0) {
            obs_mu.resize(D, Nind * K);
            obs_mean.resize(D, Nind * K);
            obs_sum.resize(D, Nind * K);
            obs_mu_sd.resize(D, Nind * K);
            obs_mu.setZero();
            obs_mean.setZero();
            obs_sum.setZero();
            obs_mu_sd.setZero();

            if (Ntrt > 1) {
                cf_mu.resize(D, Nind * K);
                cf_mean.resize(D, Nind * K);
                cf_sum.resize(D, Nind * K);
                cf_mu_sd.resize(D, Nind * K);
                cf_mu.setZero();
                cf_mean.setZero();
                cf_sum.setZero();
                cf_mu_sd.setZero();

                wald_stat.resize(D, Nind * K);
                wald_stat.setZero();
            }
        }

        TLOG("[" << std::setw(10) << (i + 1) << " / " << std::setw(10) << Nind
                 << "] found " << D << " x " << N << " <-- " << indv_name);

        Mat zz_prob = row_sub(Z, indv_index_set.at(i)); //
        zz_prob.transposeInPlace();                     // Z: K x N

        Mat zz(zz_prob.rows(), zz_prob.cols()); // K x N

        if (options.discretize) {
            zz.setZero();
            for (Index j = 0; j < zz_prob.cols(); ++j) {
                Index k;
                zz_prob.col(j).maxCoeff(&k);
                zz(k, j) += 1.0;
            }
            TLOG("Using the discretized Z");
        } else {
            zz = zz_prob;
            TLOG("Using the probabilistic Z");
        }

        ///////////////////////////////////
        // Calibrate the observed effect //
        ///////////////////////////////////
        aggregator_t obs_agg(yy, zz);
        obs_agg.verbose = options.verbose;
        obs_agg.run_gibbs(ngibbs, nburnin, a0, b0, options.log_scale);

        Mat _mu = obs_agg.mu_stat.mean().transpose();
        Mat _mu_sd = obs_agg.mu_stat.var().unaryExpr(_sqrt).transpose();

        {
            Vec _lambda = obs_agg.lambda_stat.mean();
            Vec _lambda_sd = obs_agg.lambda_stat.var().unaryExpr(_sqrt);

            for (Index j = 0; j < N; ++j) {
                const Index l = indv_index_set.at(i).at(j);
                obs_lambda(l) = _lambda(j);
                obs_lambda_sd(l) = _lambda_sd(j);
            }

            Mat _sum = yy * zz.transpose();            // D x K
            Vec _denom = zz * Mat::Ones(zz.cols(), 1); // K x 1

            for (Index k = 0; k < K; ++k) {
                out_col.push_back(indv_name + "_" + lab_name.at(k));
                obs_mu.col(s_obs) = _mu.col(k);
                obs_mu_sd.col(s_obs) = _mu_sd.col(k);

                const Scalar _denom_k = _denom(k) + eps;
                obs_mean.col(s_obs) = _sum.col(k) / _denom_k;
                obs_sum.col(s_obs) = _sum.col(k);

                ++s_obs;
            }

            TLOG("Calibrated the observed parameters");
        }

        if (Ntrt > 1) {

            /////////////////////////////////////
            // Calibrate counterfactual effect //
            /////////////////////////////////////

            std::vector<Index> cf_index;
            std::vector<Index> obs_index;

            std::tie(cf_index, obs_index) = read_cf_idx(i);

            SpMat y0 =
                read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, cf_index);

            if (options.normalize) {
                normalize_columns(y0);
                y0 *= options.col_norm;
            }

            Mat z0_prob = row_sub(Z, obs_index); //
            z0_prob.transposeInPlace();          // Z: K0 x N

            Mat z0(z0_prob.rows(), z0_prob.cols()); // K x N

            if (options.discretize) {
                z0.setZero();
                for (Index j = 0; j < z0_prob.cols(); ++j) {
                    Index k;
                    z0_prob.col(j).maxCoeff(&k);
                    z0(k, j) += 1.0;
                }
                TLOG("Using the discretized Z0");
            } else {
                z0 = z0_prob;
                TLOG("Using the probabilistic Z0");
            }

            aggregator_t agg(y0, z0);
            agg.verbose = options.verbose;
            agg.run_gibbs(ngibbs, nburnin, a0, b0, options.log_scale);

            Mat _cf_mu = agg.mu_stat.mean().transpose();
            Mat _cf_mu_sd = agg.mu_stat.var().unaryExpr(_sqrt).transpose();

            Vec _cf_lambda = agg.lambda_stat.mean();
            Vec _cf_lambda_sd = agg.lambda_stat.var().unaryExpr(_sqrt);

            for (Index j = 0; j < N; ++j) {
                const Index l = indv_index_set.at(i).at(j);
                cf_lambda(l) = _cf_lambda(j);
                cf_lambda_sd(l) = _cf_lambda_sd(j);
            }

            Mat _sum = y0 * z0.transpose();            // D x K
            Vec _denom = z0 * Mat::Ones(z0.cols(), 1); // K x 1

            /////////////////////////////////
            // Test significant divergence //
            /////////////////////////////////

            Mat _stat = (_mu - _cf_mu)
                            .binaryExpr(_mu_sd.cwiseProduct(_mu_sd) +
                                            _cf_mu_sd.cwiseProduct(_cf_mu_sd),
                                        _wald_stat_fun);

            /////////////////////
            // collect results //
            /////////////////////

            for (Index k = 0; k < K; ++k) {
                cf_mu.col(s_cf) = _cf_mu.col(k);
                cf_mu_sd.col(s_cf) = _cf_mu_sd.col(k);

                const Scalar _denom_k = _denom(k) + eps;
                cf_sum.col(s_cf) = _sum.col(k);
                cf_mean.col(s_cf) = _sum.col(k) / _denom_k;

                wald_stat.col(s_cf) = _stat.col(k);

                ++s_cf;
            }
            TLOG("Calibrated the counterfactual parameters");
        }
    }

    TLOG("Writing down the estimated effects");

    write_vector_file(output + ".cols.gz", out_col);

    write_data_file(output + ".mu.gz", obs_mu);
    write_data_file(output + ".mean.gz", obs_mean);
    write_data_file(output + ".sum.gz", obs_sum);
    write_data_file(output + ".mu_sd.gz", obs_mu_sd);

    write_data_file(output + ".lambda.gz", obs_lambda);
    write_data_file(output + ".lambda_sd.gz", obs_lambda_sd);

    if (Ntrt > 1) {
        write_data_file(output + ".cf_mu.gz", cf_mu);
        write_data_file(output + ".cf_mean.gz", cf_mean);
        write_data_file(output + ".cf_sum.gz", cf_sum);
        write_data_file(output + ".cf_mu_sd.gz", cf_mu_sd);

        write_data_file(output + ".wald.gz", wald_stat);

        write_data_file(output + ".cf_lambda.gz", cf_lambda);
        write_data_file(output + ".cf_lambda_sd.gz", cf_lambda_sd);
    }

    return EXIT_SUCCESS;
}

#endif
