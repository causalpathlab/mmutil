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
        mtx = "";
        annot_prob = "";
        ind = "";
        lab = "";
        trt_ind = "";
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
    }

    Str mtx;
    Str annot_prob;
    Str ind;
    Str trt_ind;
    Str lab;
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
        "--annot_prob (-a) : annotation/clustering probability (N x K)\n"
        "--ind (-i)        : N x 1 sample to individual (n)\n"
        "--trt_ind (-t)    : N x 1 sample to case-control membership\n"
        "--annot (-l)      : K x 1 annotation label name (e.g., cell type) \n"
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
        "${out}.sd.gz      : (M x n) SD matrix\n"
        "${out}.cols.gz    : (n x 1) Column names\n"
        "\n"
        "[Details for kNN graph]\n"
        "\n"
        "The number of bi-directional links created for every new element  \n"
        "during construction. Reasonable range for M is 2-100. Higher M work \n"
        "better on datasets with intrinsic dimensionality and/or high recall, \n"
        "while low M works better for datasets intrinsic dimensionality and/or\n"
        "low recalls. \n"
        "\n"
        "The size of the dynamic list for the nearest neighbors (used during \n"
        "the search). A higher more accurate but slower search. This cannot be\n"
        "set lower than the number nearest neighbors k. The value ef of can be \n"
        "anything between of the dataset. [Reference] Malkov, Yu, and Yashunin. "
        "\n"
        "`Efficient and robust approximate nearest neighbor search using \n"
        "Hierarchical Navigable Small World graphs.` \n"
        "\n"
        "preprint: "
        "https://arxiv.org/abs/1603.09320 \n"
        "See also: https://github.com/nmslib/hnswlib"
        "\n";

    const char *const short_opts = "m:a:i:l:t:o:LRB:r:u:w:g:G:DPC:k:b:n:hv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "annot_prob", required_argument, nullptr, 'a' }, //
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
        case 'a':
            options.annot_prob = std::string(optarg);
            break;
        case 'i':
            options.ind = std::string(optarg);
            break;
        case 't':
            options.trt_ind = std::string(optarg);
            break;
        case 'l':
            options.lab = std::string(optarg);
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
    ERR_RET(!file_exists(options.annot_prob), "No ANNOT_PROB data file");
    ERR_RET(!file_exists(options.ind), "No IND data file");
    ERR_RET(!file_exists(options.lab), "No LAB data file");

    ERR_RET(options.rank < 2, "Too small rank");

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
            // for (auto x : cf_idx) {
            //     std::cout << x << " ";
            // }
            // std::cout << std::endl;
        }
    }
};

template <typename OPTIONS>
int
aggregate_col(const OPTIONS &options)
{

    const std::string mtx_file = options.mtx;
    const std::string idx_file = options.mtx + ".index";
    const std::string annot_prob_file = options.annot_prob;
    const std::string ind_file = options.ind;
    const std::string lab_file = options.lab;
    const Index ngibbs = options.ngibbs;
    const Index nburnin = options.nburnin;
    const std::string row_weight_file = options.row_weight_file;
    const std::string output = options.out;

    Mat Z;
    CHECK(read_data_file(annot_prob_file, Z));
    TLOG("Latent membership matrix: " << Z.rows() << " x " << Z.cols());

    const Index K = Z.cols();

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

    const Index Nsample = indv.size();
    const Index Nind = indv_id_name.size();

    TLOG("Identified " << Nind << " individuals");

    /////////////////
    // label names //
    /////////////////

    std::vector<std::string> lab_name;
    lab_name.reserve(K);
    CHECK(read_vector_file(lab_file, lab_name));

    ASSERT(lab_name.size() == K,
           "Need the same number of label names for the columns of Z");

    TLOG("Identified " << K << " labels");

    ASSERT(Z.rows() == Nsample, "rows(Z) != Nsample");

    ///////////////////////////////////////
    // case-control treatment membership //
    ///////////////////////////////////////

    std::vector<std::string> trt_membership;
    trt_membership.reserve(Nsample);

    std::vector<std::string> trt_id_name;
    std::vector<Index> trt; // map: col -> trt index

    if (file_exists(options.trt_ind)) {

        ////////////////////////
        // read from the file //
        ////////////////////////

        CHECK(read_vector_file(options.trt_ind, trt_membership));
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
#ifdef DEBUG
            TLOG("[" << lb << ", " << ub << ")");
#endif
            std::copy(col_k.begin() + lb, col_k.begin() + ub, subcol_k.begin());

            Mat x0 = read_y_block(subcol_k);

#ifdef DEBUG
            ASSERT(x0.cols() == subcol_k.size(), "size doesn't match");
#endif

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

    Mat out_mu;
    Mat out_mu_sd;

    Mat cf_mu;
    Mat cf_mu_sd;

    std::vector<std::string> out_col;

    /**
     * @param i individual index [0, Nind)
     */
    auto read_y = [&](const Index i) {
        using namespace mmutil::index;
        return read_eigen_sparse_subset_col(mtx_file,
                                            mtx_idx_tab,
                                            indv_index_set.at(i));
    };

    /**
     * @param i individual index [0, Nind)
     */
    auto read_y_cf = [&](const Index i) {
        using namespace mmutil::index;
        std::vector<Index> cf_indv_index;
        cf_indv_index.reserve(indv_index_set.at(i).size());
        float *mass = V.data();

        for (Index j : indv_index_set.at(i)) {
            const Index tj = trt.at(j);
            const Index sj = cf_index_sampler(tj);
            KnnAlg &alg = *knn_lookup_vec[sj].get();

            const Index n_sj = trt_index_set.at(sj).size();
            const std::size_t nquery = std::min(options.knn, n_sj);

            auto pq = alg.searchKnn((void *)(mass + rank * j), nquery);
            std::size_t k;

            while (!pq.empty()) {
                std::tie(std::ignore, k) = pq.top();
                cf_indv_index.emplace_back(trt_index_set.at(sj).at(k));
                pq.pop();
            }
        }

        return read_eigen_sparse_subset_col(mtx_file,
                                            mtx_idx_tab,
                                            cf_indv_index);
    };

    /**
     * @param i individual index [0, Nind)
     */
    auto read_z = [&](const Index i) {
        return row_sub(Z, indv_index_set.at(i));
    };

    auto _sqrt = [](const Scalar &x) -> Scalar {
        return (x > 0.) ? std::sqrt(x) : 0.;
    };

    Index s_obs = 0; // cumulative for obs (be cautious; do not touch)
    Index s_cf = 0;  // cumulative for cf (be cautious; do not touch)

    for (Index i = 0; i < Nind; ++i) {

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted at Ind = " << (i));
            return EXIT_FAILURE;
        }
#endif

        const std::string indv_name = indv_id_name.at(i);

        // Y: features x columns
        SpMat yy = read_y(i);
        const Index D = yy.rows();
        const Index N = yy.cols();

        // Ycf: counter-factual data by matching
        SpMat y0;
        if (Ntrt > 1) {
            y0 = read_y_cf(i);
        }

        if (i == 0) {
            out_mu.resize(D, Nind * K);
            out_mu_sd.resize(D, Nind * K);
            out_mu.setZero();
            out_mu_sd.setZero();

            if (Ntrt > 1) {
                cf_mu.resize(D, Nind * K);
                cf_mu_sd.resize(D, Nind * K);
                cf_mu.setZero();
                cf_mu_sd.setZero();
            }
        }

        TLOG("[" << std::setw(10) << (i + 1) << " / " << std::setw(10) << Nind
                 << "] found " << D << " x " << N << " <-- " << indv_name);

        Mat zz_prob = read_z(i);    // Z: type x columns
        zz_prob.transposeInPlace(); //

        Mat zz(zz_prob.rows(), zz_prob.cols()); // type x columns

        if (options.discretize) {
            TLOG("Using a discretized annotation matrix Z");
            zz.setZero();
            for (Index j = 0; j < zz_prob.cols(); ++j) {
                Index k;
                zz_prob.col(j).maxCoeff(&k);
                zz(k, j) += 1.0;
            }
        } else {
            TLOG("Using a probabilistic annotation matrix Z");
            zz = zz_prob;
        }

        ///////////////////////////////////
        // Calibrate the observed effect //
        ///////////////////////////////////

        {
            aggregator_t agg(yy, zz);
            agg.verbose = options.verbose;
            agg.run_gibbs(ngibbs, nburnin);

            Mat _mean = agg.mu_stat.mean().transpose();
            Mat _sd = agg.mu_stat.var().unaryExpr(_sqrt).transpose();

            for (Index k = 0; k < K; ++k) {
                out_col.push_back(indv_name + "_" + lab_name.at(k));
                out_mu.col(s_obs) = _mean.col(k);
                out_mu_sd.col(s_obs) = _sd.col(k);
                ++s_obs;
            }
        }

        TLOG("Calibrated the observed parameters");

        /////////////////////////////////////
        // Calibrate counterfactual effect //
        /////////////////////////////////////

        if (Ntrt > 1) {
            aggregator_t agg(y0, zz);
            agg.verbose = options.verbose;
            agg.run_gibbs(ngibbs, nburnin);

            Mat _mean = agg.mu_stat.mean().transpose();
            Mat _sd = agg.mu_stat.var().unaryExpr(_sqrt).transpose();

            for (Index k = 0; k < K; ++k) {
                cf_mu.col(s_cf) = _mean.col(k);
                cf_mu_sd.col(s_cf) = _sd.col(k);
                ++s_cf;
            }
            TLOG("Calibrated the counterfactual parameters");
        }
    }

    TLOG("Writing down the estimated effects");

    write_vector_file(output + ".cols.gz", out_col);
    write_data_file(output + ".mean.gz", out_mu);
    write_data_file(output + ".sd.gz", out_mu_sd);

    if (Ntrt > 1) {
        write_data_file(output + ".cf_mean.gz", cf_mu);
        write_data_file(output + ".cf_sd.gz", cf_mu_sd);
    }

    return EXIT_SUCCESS;
}

#endif
