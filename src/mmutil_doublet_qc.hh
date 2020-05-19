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
        prob_file = "";
        lab_file = "";
        out = "output";
        row_weight_file = "";

        tau = 1.0;
        rank = 50;
        lu_iter = 5;
        col_norm = 1000;

        raw_scale = true;
        log_scale = false;

        knn = 50;
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
    }

    Str mtx;
    Str col;
    Str row;
    Str prob_file;
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
        "--col (-c)             : data column file\n"
        "--feature (-f)         : data row file (features)\n"
        "--row (-f)             : data row file (features)\n"
        "--assign (-p)          : membership assignment probability file\n"
        "--prob (-p)            : membership assignment probability file\n"
        "--label (-q)           : label name file\n"
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
        "--verbose (-v)         : Set verbose (default: false)\n"
        "\n";

    const char *const short_opts = "m:c:f:p:q:r:l:C:w:k:b:n:o:B:LRi:t:hv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "col", required_argument, nullptr, 'c' },        //
          { "row", required_argument, nullptr, 'f' },        //
          { "feature", required_argument, nullptr, 'f' },    //
          { "prob", required_argument, nullptr, 'p' },       //
          { "argmax", required_argument, nullptr, 'p' },     //
          { "lab", required_argument, nullptr, 'q' },        //
          { "label", required_argument, nullptr, 'q' },      //
          { "rank", required_argument, nullptr, 'r' },       //
          { "lu_iter", required_argument, nullptr, 'l' },    //
          { "row_weight", required_argument, nullptr, 'w' }, //
          { "col_norm", required_argument, nullptr, 'C' },   //
          { "bilink", required_argument, nullptr, 'b' },     //
          { "nlist", required_argument, nullptr, 'n' },      //
          { "out", required_argument, nullptr, 'o' },        //
          { "log_scale", no_argument, nullptr, 'L' },        //
          { "raw_scale", no_argument, nullptr, 'R' },        //
          { "batch_size", required_argument, nullptr, 'B' }, //
          { "em_iter", required_argument, nullptr, 'i' },    //
          { "em_tol", required_argument, nullptr, 't' },     //
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

        case 'p':
            options.prob_file = std::string(optarg);
            break;

        case 'q':
            options.lab_file = std::string(optarg);
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
    ERR_RET(!file_exists(options.prob_file), "No probability file");
    ERR_RET(!file_exists(options.lab_file), "No label file");
    ERR_RET(options.rank < 2, "Too small rank");

    return EXIT_SUCCESS;
}

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

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index N = info.max_col;

    TLOG("N = " << N);

    std::vector<mmutil::index::idx_pair_t> idx_tab;
    CHECK(mmutil::index::read_mmutil_index(idx_file, idx_tab));

    ////////////////////////////////////
    // cell-level assignment results  //
    ////////////////////////////////////

    Mat Pr; // N x K assignment probability file
    CHECK(read_data_file(options.prob_file, Pr));
    const Index K = Pr.cols();
    CHK_ERR_RET(Pr.rows() != N,
                "Must have N x K matrix: " << Pr.rows() << " vs. " << N);
    Pr.transposeInPlace();

    std::vector<Index> membership(N);
    Index argmax = 0;
    for (Index i = 0; i < N; ++i) {
        Pr.col(i).maxCoeff(&argmax);
        membership[i] = argmax;
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

    std::vector<std::string> labels;
    CHK_ERR_RET(read_vector_file(options.lab_file, labels),
                "Failed to read the label file: " << options.lab_file);

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

        TLOG(n_singlet << " singlets vs. " << n_doublet << " doublets");

        hnswlib::InnerProductSpace VS(rank);
        KnnAlg alg(&VS, n_singlet + n_doublet, param_bilink, param_nnlist);
        Index cum_idx = 0;

        {
            // adding singlet points
            float *mass = ss.data();
            const Index vecdim = ss.rows();
            const Index vecsize = n_singlet;

            //#pragma omp parallel for
            for (Index j = 0; j < vecsize; ++j) {
                alg.addPoint((void *)(mass + vecdim * j), cum_idx);
                ++cum_idx;
            }
        }

        {
            // adding doublet points
            float *mass = dd.data();
            const Index vecdim = dd.rows();
            const Index vecsize = n_doublet;

            //#pragma omp parallel for
            for (Index j = 0; j < vecsize; ++j) {
                alg.addPoint((void *)(mass + vecdim * j), cum_idx);
                ++cum_idx;
            }
        }

        TLOG("Added " << cum_idx << " points");
        std::vector<std::tuple<Index, Index, Scalar>> _index;
        const Index n_tot = n_singlet + n_doublet;
        const Index nquery = (n_tot) > knn ? knn : (n_tot - 1);

        // TLOG
        {
            float *mass = ss.data();
            const Index d = ss.rows();
            progress_bar_t<Index> prog(n_singlet, 1e2);
            for (Index i = 0; i < n_singlet; ++i) {
                auto pq = alg.searchKnn((void *)(mass + d * i), nquery);
                float d = 0;
                std::size_t k;
                while (!pq.empty()) {
                    std::tie(d, k) = pq.top();
                    _index.emplace_back(i, k, 1. - d);
                    pq.pop();
                }
                prog.update();
                prog(std::cerr);
            }
        }

        {
            float *mass = dd.data();
            const Index d = dd.rows();
            progress_bar_t<Index> prog(n_doublet, 1e2);
            for (Index i = n_singlet; i < n_tot; ++i) {
                std::size_t j = (i - n_singlet);
                auto pq = alg.searchKnn((void *)(mass + d * j), nquery);
                float d = 0;
                std::size_t k;
                while (!pq.empty()) {
                    std::tie(d, k) = pq.top();
                    _index.emplace_back(i, k, 1. - d);
                    pq.pop();
                }
                prog.update();
                prog(std::cerr);
            }
        }

        TLOG("Constructed KNN graph");
        auto knn_index = keep_reciprocal_knn(_index, true);

        TLOG("Reciprocal matching KNN graph");
        auto lc_ptr = build_lc_model(knn_index, K);
        auto &lc = *lc_ptr.get();

        update_latent_random(lc, 1, K); // assign

        std::unordered_set<Index> clamp;

        {
            Index e = 0;       // We want to clamp membership for
            const Index k = 0; // the edges linking singlets
            for (auto &tt : knn_index) {
                Index i = std::get<0>(tt), j = std::get<1>(tt);
                if (i < n_singlet && j < n_singlet) {
                    lc.Z.col(e).setZero();
                    lc.Z(k, e) = 1.;
                    clamp.insert(e);
                }
                ++e;
            }
        }

        const Index nlocal = options.lc_nlocal;
        const Index ngibbs = options.lc_ngibbs;
        const Index nburnin = options.lc_nburnin;

        Mat mu, sd, mu_vb, sd_vb;
        std::tie(mu, sd) = run_gibbs_sampling(lc,
                                              clamp,
                                              ngibbs_t(ngibbs),
                                              nburnin_t(nburnin),
                                              nlocal_t(nlocal));

        std::tie(mu_vb, sd_vb) =
            run_vb_optimization(lc, nvb_t(ngibbs), nlocal_t(nlocal));

        std::string outhdr = options.out + "." + lab_k;

        write_tuple_file(outhdr + ".network.gz", knn_index);
        write_data_file(outhdr + ".prop.gz", mu);
        write_data_file(outhdr + ".sd.gz", mu);
        write_data_file(outhdr + ".prop_vb.gz", mu_vb);
        write_data_file(outhdr + ".sd_vb.gz", mu_vb);
    }

    return EXIT_SUCCESS;
}

#endif
