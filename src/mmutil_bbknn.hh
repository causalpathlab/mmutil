#include <getopt.h>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "eigen_util.hh"
#include "io.hh"
#include "mmutil.hh"
#include "mmutil_index.hh"
#include "mmutil_normalize.hh"
#include "mmutil_stat.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "ext/tabix/bgzf.h"

#include "std_util.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_BBKNN_HH_
#define MMUTIL_BBKNN_HH_

struct bbknn_options_t {

    using Str = std::string;

    bbknn_options_t()
    {
        mtx = "";
        col = "";
        batch = "";
        out = "output";
        verbose = false;

        tau = 1.0;
        rank = 50;
        lu_iter = 5;
        knn = 50;
        bilink = 5; // 2 ~ 100 (bi-directional link per element)
        nlist = 51; // knn ~ N (nearest neighbour)

        raw_scale = true;
        log_scale = false;

        col_norm = 10000;
        block_size = 5000;

        em_iter = 10;
        em_tol = 1e-2;
    }

    Str mtx;
    Str col;
    Str batch;
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

    // link community
    Index lc_ngibbs;
    Index lc_nlocal;
    Index lc_nburnin;

    bool verbose;
};

template <typename OPTIONS>
int
parse_bbknn_options(const int argc, const char *argv[], OPTIONS &options)
{

    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)        : data MTX file (M x N)\n"
        "--data (-m)       : data MTX file (M x N)\n"
        "--col (-c)        : data col file (samples)\n"
        "--batch (-t)      : N x 1 batch assignment file (e.g., individuals) \n"
        "--out (-o)        : Output file header\n"
        "\n"
        "[Matching options]\n"
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
        "${out}.mtx.gz    : (N x N) adjacency matrix\n"
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

    const char *const short_opts = "m:c:t:o:LRB:r:u:w:C:k:b:n:G:A:U:hv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "col", required_argument, nullptr, 'c' },        //
          { "batch", required_argument, nullptr, 't' },      //
          { "out", required_argument, nullptr, 'o' },        //
          { "log_scale", no_argument, nullptr, 'L' },        //
          { "raw_scale", no_argument, nullptr, 'R' },        //
          { "block_size", required_argument, nullptr, 'B' }, //
          { "rank", required_argument, nullptr, 'r' },       //
          { "lu_iter", required_argument, nullptr, 'u' },    //
          { "row_weight", required_argument, nullptr, 'w' }, //
          { "col_norm", required_argument, nullptr, 'C' },   //
          { "knn", required_argument, nullptr, 'k' },        //
          { "bilink", required_argument, nullptr, 'b' },     //
          { "nlist", required_argument, nullptr, 'n' },      //
          { "gibbs", required_argument, nullptr, 'G' },      //
          { "ngibbs", required_argument, nullptr, 'G' },     //
          { "local", required_argument, nullptr, 'A' },      //
          { "nlocal", required_argument, nullptr, 'A' },     //
          { "burnin", required_argument, nullptr, 'U' },     //
          { "nburnin", required_argument, nullptr, 'U' },    //
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

        case 't':
            options.batch = std::string(optarg);
            break;

        case 'o':
            options.out = std::string(optarg);
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

        case 'B':
            options.block_size = std::stoi(optarg);
            break;

        case 'b':
            options.bilink = std::stoi(optarg);
            break;

        case 'n':
            options.nlist = std::stoi(optarg);
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
    ERR_RET(!file_exists(options.batch), "No batch data file");
    ERR_RET(options.rank < 2, "Too small rank");
    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
build_bbknn(const OPTIONS &options)
{

    const std::string mtx_file = options.mtx;
    const std::string idx_file = options.mtx + ".index";
    const std::string col_file = options.col;
    const std::string batch_file = options.batch;

    const std::string row_weight_file = options.row_weight_file;
    const std::string output = options.out;

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
    const Index Nsample = info.max_col;

    std::vector<std::string> columns;
    CHECK(read_vector_file(col_file, columns));

    ASSERT(Nsample == columns.size(),
           "The number of columns mismatch between col and mtx");

    //////////////////////
    // batch membership //
    //////////////////////

    std::vector<std::string> batch_membership;
    batch_membership.reserve(Nsample);
    CHECK(read_vector_file(batch_file, batch_membership));

    ASSERT(batch_membership.size() == Nsample,
           "This batch membership file mismatches with mtx data");

    std::vector<std::string> batch_id_name;
    std::vector<Index> batch; // map: col -> batch index

    std::tie(batch, batch_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(batch_membership);

    auto batch_index_set = make_index_vec_vec(batch);

    ASSERT(batch.size() >= Nsample, "Need batch membership for each column");
    const Index Nbatch = batch_id_name.size();
    TLOG("Identified " << Nbatch << " batches");

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

    ////////////////////////////////
    // Learn latent embedding ... //
    ////////////////////////////////

    TLOG("Training SVD for spectral matching ...");
    svd_out_t svd = take_svd_online_em(mtx_file, idx_file, ww, options);
    proj.resize(svd.U.rows(), svd.U.cols());
    proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
    TLOG("Found projection: " << proj.rows() << " x " << proj.cols());

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
        std::vector<Index> &col_k = batch_index_set[k];
        const Index Nk = col_k.size();
        const Index block_size = options.block_size;
        const Index rank = proj.cols();

        Mat ret(rank, Nk);
        ret.setZero();

        Index r = 0;
        for (Index lb = 0; lb < Nk; lb += block_size) {
            const Index ub = std::min(Nk, block_size + lb);

            std::vector<Index> subcol_k(ub - lb);
            std::copy(col_k.begin() + lb, col_k.begin() + ub, subcol_k.begin());

            Mat x0 = read_y_block(subcol_k);

            Mat xx = make_normalized_laplacian(x0,
                                               ww,
                                               options.tau,
                                               options.col_norm,
                                               options.log_scale);

            Mat vv = proj.transpose() * xx; // rank x block_size
            normalize_columns(vv);

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

    /////////////////////////////////////////
    // construct dictionary for each batch //
    /////////////////////////////////////////

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_vec;
    Mat V(rank, Nsample);

    for (Index bb = 0; bb < Nbatch; ++bb) {
        const Index n_tot = batch_index_set[bb].size();

        using vs_type = hnswlib::InnerProductSpace;

        vs_vec.push_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec[vs_vec.size() - 1].get();
        knn_lookup_vec.push_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    {
        progress_bar_t<Index> prog(Nsample, 1e2);

        for (Index bb = 0; bb < Nbatch; ++bb) {
            const Index n_tot = batch_index_set[bb].size();
            KnnAlg &alg = *knn_lookup_vec[bb].get();
            Mat dat = build_spectral_data(bb);
            float *mass = dat.data(); // adding data points
            for (Index i = 0; i < n_tot; ++i) {
                alg.addPoint((void *)(mass + rank * i), i);
                const Index j = batch_index_set.at(bb).at(i);
                V.col(j) = dat.col(i);
                prog.update();
                prog(std::cerr);
            }
        }
    }

    ///////////////////////
    // For each column j //
    ///////////////////////

    std::vector<std::tuple<Index, Index, Scalar>> knn_index;

    float *mass = V.data();
    std::vector<Scalar> dist_j(options.knn);
    std::vector<Scalar> weights_j(options.knn);
    std::vector<Index> neigh_j(options.knn);

    progress_bar_t<Index> prog(Nsample, 1e2);

    for (Index j = 0; j < Nsample; ++j) {

        Index deg_j = 0;

        for (Index bb = 0; bb < Nbatch; ++bb) {
            KnnAlg &alg = *knn_lookup_vec[bb].get();
            Index nn_b = batch_index_set.at(bb).size();
            std::size_t nquery = (std::min(options.knn, nn_b) / Nbatch);
            if (nquery < 1)
                nquery = 1;

            auto pq = alg.searchKnn((void *)(mass + rank * j), nquery);
            while (!pq.empty()) {
                float d = 0;
                std::size_t k;
                std::tie(d, k) = pq.top();
                Index i = batch_index_set.at(bb).at(k);
                if (i != j) {
                    if (deg_j < dist_j.size()) {
                        dist_j[deg_j] = d;
                        neigh_j[deg_j] = i;
                        ++deg_j;
                    }
                }
                pq.pop();
            }
        }

        normalize_weights(deg_j, dist_j, weights_j);

        for (Index i = 0; i < deg_j; ++i) {
            const Index k = neigh_j[i];
            const Scalar w = weights_j[i];
            ASSERT(w > .0, "must be non-negative");
            knn_index.emplace_back(j, k, w);
        }
        prog.update();
        prog(std::cerr);
    }

    const SpMat A = build_eigen_sparse(knn_index, Nsample, Nsample);
    write_matrix_market_file(options.out + ".mtx.gz", A);

    return EXIT_SUCCESS;
}

#endif
