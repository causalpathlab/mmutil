#include <getopt.h>
#include <unordered_map>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "utils/progress.hh"
#include "mmutil_pois.hh"

#ifndef MMUTIL_CFA_COL_HH_
#define MMUTIL_CFA_COL_HH_

struct cfa_options_t {

    cfa_options_t()
    {
        mtx_file = "";
        annot_prob_file = "";
        annot_file = "";
        ind_file = "";
        annot_name_file = "";
        trt_ind_file = "";
        out = "output";
        verbose = false;

        tau = 1.0;   // Laplacian regularization
        rank = 10;   // SVD rank
        lu_iter = 5; // randomized SVD
        knn = 1;     // k-nearest neighbours
        bilink = 5;  // 2 ~ 100 (bi-directional link per element)
        nlist = 5;   // knn ~ N (nearest neighbour)

        raw_scale = true;
        log_scale = false;

        col_norm = 1000;
        block_size = 5000;

        em_iter = 10;
        em_tol = 1e-2;

        gamma_a0 = 1;
        gamma_b0 = 1;

        discretize = true;
        normalize = false;
    }

    std::string mtx_file;
    std::string annot_prob_file;
    std::string annot_file;
    std::string col_file;
    std::string ind_file;
    std::string trt_ind_file;
    std::string annot_name_file;
    std::string out;

    // SVD and matching
    std::string row_weight_file;

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

    // For Bayesian calibration and Wald stat
    // Scalar wald_reg;
    Scalar gamma_a0;
    Scalar gamma_b0;

    bool verbose;

    // pois
    bool discretize;
    bool normalize;
};

///////////////////////////////////////////////
// Estimate sequencing depth given mu matrix //
///////////////////////////////////////////////

struct cfa_depth_finder_t {

    explicit cfa_depth_finder_t(const Mat &_mu,
                                const Mat &_zz,
                                const std::vector<Index> &_indv,
                                const Scalar a0,
                                const Scalar b0)
        : Mu(_mu)
        , Z(_zz)
        , indv(_indv)
        , D(Mu.rows())
        , K(Z.cols())
        , Nsample(Z.rows())
        , opt_op(a0, b0)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const Index r, const Index c, const Index e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        num_vec.resize(max_col, 1);
        denom_vec.resize(max_col, 1);
        num_vec.setZero();
        denom_vec.setZero();
    }

    void eval(const Index row, const Index col, const Scalar weight)
    {

        if (row < max_row && col < max_col) {

            const Index ii = indv.at(col);
            Scalar num = 0., denom = 0.;

            num += weight;

            for (Index k = 0; k < K; ++k) { // [ii * K, (ii+1)*K)
                const Index j = ii * K + k;
                const Scalar z_k = Z(col, k);
                denom += Mu(row, j) * z_k;
            }

            num_vec(col) += num;
            denom_vec(col) += denom;
        }
#ifdef DEBUG
        else {
            TLOG("[" << row << ", " << col << ", " << weight << "]");
            TLOG(max_row << " x " << max_col);
        }
#endif
    }

    void eval_end_of_file() {}

    Vec estimate_depth() { return num_vec.binaryExpr(denom_vec, opt_op); }

    const Mat &Mu;                  // D x (K * Nind)
    const Mat &Z;                   // Nsample x K
    const std::vector<Index> &indv; // Nsample x 1
    const Index D;
    const Index K;
    const Index Nsample;

private:
    BGZF *fp;

    Index max_row;
    Index max_col;
    Index max_elem;

    Vec num_vec;
    Vec denom_vec;

private:
    poisson_t::rate_opt_op_t opt_op;
};

////////////////////////////////
// Adjust confounding factors //
////////////////////////////////

struct cfa_normalizer_t {

    explicit cfa_normalizer_t(const Mat &_mu,
                              const Mat &_zz,
                              const Vec &_rho,
                              const std::vector<Index> &_indv,
                              const std::string _outfile)
        : Mu(_mu)
        , Z(_zz)
        , rho(_rho)
        , indv(_indv)
        , D(Mu.rows())
        , K(Z.cols())
        , Nsample(Z.rows())
        , outfile(_outfile)
    {
        ASSERT(Z.rows() == indv.size(),
               "Needs the annotation and membership for each column");
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const Index r, const Index c, const Index e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        // confirm that the sizes are compatible
        ASSERT(D == r, "dimensionality should match");
        ASSERT(Nsample == c, "sample size should match");
        ofs.open(outfile.c_str(), std::ios::out);
        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << max_row << FS << max_col << FS << max_elem << std::endl;
        elem_check = 0;
    }

    void eval(const Index row, const Index col, const Scalar weight)
    {
        const Index ii = indv.at(col);

        Scalar denom = 0.;

        for (Index k = 0; k < K; ++k) { // [ii * K, (ii+1)*K)
            const Index j = ii * K + k;
            const Scalar z_k = Z(col, k);
            denom += Mu(row, j) * z_k;
        }

        // weight <- weight / denom;
        if (row < max_row && col < max_col) {
            const Index i = row + 1; // fix zero-based to one-based
            const Index j = col + 1; // fix zero-based to one-based

            if (denom > 0. && rho(col) > 0.) {
                const Scalar new_weight = weight / denom / rho(col);
                ofs << i << FS << j << FS << new_weight << std::endl;
            } else {
                ofs << i << FS << j << FS << weight << std::endl;
            }
            elem_check++;
        }
    }

    void eval_end_of_file()
    {
        ofs.close();
        ASSERT(max_elem == elem_check, "Failed to write all the elements");
    }

    const Mat &Mu;                  // D x (K * Nind)
    const Mat &Z;                   // Nsample x K
    const Vec &rho;                 // Nsample x 1
    const std::vector<Index> &indv; // Nsample x 1
    const Index D;
    const Index K;
    const Index Nsample;

    const std::string outfile;

private:
    obgzf_stream ofs;
    BGZF *fp;
    Index max_row;
    Index max_col;
    Index max_elem;
    Index elem_check;
    static constexpr char FS = ' ';
};

template <typename OPTIONS>
int
cfa_col(const OPTIONS &options)
{

    using namespace mmutil::io;
    using namespace mmutil::index;

    const std::string mtx_file = options.mtx_file;
    const std::string idx_file = options.mtx_file + ".index";
    const std::string annot_prob_file = options.annot_prob_file;
    const std::string annot_file = options.annot_file;
    const std::string col_file = options.col_file;
    const std::string ind_file = options.ind_file;
    const std::string annot_name_file = options.annot_name_file;
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

    std::vector<std::string> annot_name;
    CHECK(read_vector_file(annot_name_file, annot_name));
    auto lab_position = make_position_dict<std::string, Index>(annot_name);
    const Index K = annot_name.size();

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

    ASSERT(annot_name.size() == Z.cols(),
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
    std::vector<Index> trt_map; // map: col -> trt index

    TLOG("Found treatment membership file: " << options.trt_ind_file);

    CHECK(read_vector_file(options.trt_ind_file, trt_membership));

    ASSERT(trt_membership.size() == Z.rows(),
           "size(Treatment) != row(Z) " << trt_membership.size() << " vs. "
                                        << Z.rows());

    std::tie(trt_map, trt_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(trt_membership);

    auto trt_index_set = make_index_vec_vec(trt_map);
    const Index Ntrt = trt_index_set.size();
    TLOG("Identified " << Ntrt << " treatment conditions");

    ASSERT(Ntrt > 1, "Must have more than one treatment conditions");

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHECK(build_mmutil_index(mtx_file, idx_file));

    CHECK(read_mmutil_index(idx_file, mtx_idx_tab));

    CHECK(check_index_tab(mtx_file, mtx_idx_tab));

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;

    ASSERT(Nsample == info.max_col,
           "Should have matched .mtx.gz, N = " << Nsample << " vs. "
                                               << info.max_col);

    ///////////////////////////////////
    // weights for the rows/features //
    ///////////////////////////////////

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> _ww;
        CHECK(read_vector_file(row_weight_file, _ww));
        weights = eigen_vector(_ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    ////////////////////////////////
    // Learn latent embedding ... //
    ////////////////////////////////

    TLOG("Training SVD for spectral matching ...");
    svd_out_t svd = take_svd_online_em(mtx_file, idx_file, ww, options);
    TLOG("Done SVD");
    Mat proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
    TLOG("Found projection: " << proj.rows() << " x " << proj.cols());

    /// Take a block of Y matrix
    /// @param subcol cells
    /// @returns y

    auto read_y_block = [&](const std::vector<Index> &subcol) -> Mat {
        return Mat(read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, subcol));
    };

    /// Read the annotation matrix
    /// @param subcol cells
    /// @returns z

    auto read_z_block = [&](const std::vector<Index> &subcol) -> Mat {
        Mat zz_prob = row_sub(Z, subcol); //
        zz_prob.transposeInPlace();       // Z: K x N

        Mat zz(zz_prob.rows(), zz_prob.cols()); // K x N

        if (options.discretize) {
            zz.setZero();
            for (Index j = 0; j < zz_prob.cols(); ++j) {
                Index k;
                zz_prob.col(j).maxCoeff(&k);
                zz(k, j) += 1.0;
            }
            // TLOG("Using the discretized Z: " << zz.sum());
        } else {
            zz = zz_prob;
            // TLOG("Using the probabilistic Z: " << zz.sum());
        }
        return zz;
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

    ///////////////////////////////////////////////////////
    // construct dictionary for each treatment condition //
    ///////////////////////////////////////////////////////
    using vs_type = hnswlib::InnerProductSpace;

    // std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec_trt;
    // std::vector<std::shared_ptr<KnnAlg>> knn_lookup_trt;

    // TLOG("Constructing spectral dictionary for matching");

    // for (Index tt = 0; tt < Ntrt; ++tt) {
    //     const Index n_tot = trt_index_set[tt].size();

    //     vs_vec_trt.emplace_back(std::make_shared<vs_type>(rank));

    //     vs_type &VS = *vs_vec_trt[tt].get();
    //     knn_lookup_trt.emplace_back(
    //         std::make_shared<KnnAlg>(&VS, n_tot, param_bilink,
    //         param_nnlist));
    // }

    //////////////////////////////////////////////
    // construct dictionary for each individual //
    //////////////////////////////////////////////

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec_indv;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_indv;

    for (Index ii = 0; ii < Nind; ++ii) {
        const Index n_tot = indv_index_set[ii].size();
        vs_vec_indv.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_indv[ii].get();
        knn_lookup_indv.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    /////////////////////
    // add data points //
    /////////////////////

    Mat V(rank, Nsample);

    const Index block_size = options.block_size;

    for (Index lb = 0; lb < Nsample; lb += block_size) {
        const Index ub = std::min(Nsample, block_size + lb);

        std::vector<Index> sub_b(ub - lb);
        std::iota(sub_b.begin(), sub_b.end(), lb);

        Mat x0 = read_y_block(sub_b);

        Mat xx = make_normalized_laplacian(x0,
                                           ww,
                                           options.tau,
                                           options.col_norm,
                                           options.log_scale);

        Mat vv = proj.transpose() * xx; // rank x block_size
        normalize_columns(vv);          // to make cosine distance

        for (Index j = 0; j < vv.cols(); ++j) {
            const Index r = sub_b[j];
            V.col(r) = vv.col(j);
        }
    }

    TLOG("Updating each individual's dictionary");

    progress_bar_t<Index> prog(Nsample, 1e2);

    for (Index ii = 0; ii < Nind; ++ii) {
        const Index n_tot = indv_index_set[ii].size(); // # cells
        KnnAlg &alg = *knn_lookup_indv[ii].get();      // lookup
        float *mass = V.data();                        // raw data

        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = indv_index_set.at(ii).at(i);
            alg.addPoint((void *)(mass + rank * cell_j), i);
            prog.update();
            prog(std::cerr);
        }
    }

    ///////////////////////////
    // For each individual i //
    ///////////////////////////

    const std::size_t knn_each = options.knn;
    const std::size_t knn_max = knn_each * (Nind - 1);

    std::vector<Scalar> within_dist(knn_max), within_weights(knn_max);
    std::vector<Index> within_neigh(knn_max);

    std::vector<Scalar> between_dist(knn_max), between_weights(knn_max);
    std::vector<Index> between_neigh(knn_max);

    /// Read counterfactually-matched blocks
    /// @param ind_i individual i [0, Nind)
    /// @returns (y0_within, z0_within, y0_between, z0_between)
    /// D x n_i and K x n_i
    auto read_cf_block = [&](const std::vector<Index> &cells_j) {
        float *mass = V.data();
        const Index n_j = cells_j.size();

        Mat y0_within(D, n_j), z0_within(K, n_j);
        Mat y0_between(D, n_j), z0_between(K, n_j);
        y0_within.setZero();
        z0_within.setZero();
        y0_between.setZero();
        z0_between.setZero();

        // TLOG("Constructing the counterfactual data of "
        //      << n_j << " cells, max knn=" << knn_max);

        for (Index jth = 0; jth < n_j; ++jth) {    // For each cell j
            const Index _cell_j = cells_j.at(jth); //
            const Index tj = trt_map.at(_cell_j);  // Trt group for this cell j
            const Index jj = indv.at(_cell_j);     // Individual for this cell j
            Index within_deg = 0;                  // # of neighbours
            Index between_deg = 0;                 // # of neighbours

            for (Index ii = 0; ii < Nind; ++ii) { // Pick cells from the other
                if (ii == jj)                     // individuals
                    continue;                     // skip the same individual

                const std::vector<Index> &cells_i = indv_index_set.at(ii);

                KnnAlg &alg = *knn_lookup_indv[ii].get();
                const std::size_t n_i = cells_i.size();
                const std::size_t nquery = std::min(knn_each, n_i);

                auto pq =
                    alg.searchKnn((void *)(mass + rank * _cell_j), nquery);

                while (!pq.empty()) {
                    float d = 0;                              // distance
                    std::size_t k;                            // local index
                    std::tie(d, k) = pq.top();                //
                    const Index _cell_i = cells_i.at(k);      // global index
                    const Index ti = trt_map.at(_cell_i);     // treatment
                                                              //
                    if (ti == tj) {                           // matching trt
                        within_dist[within_deg] = d;          //
                        within_neigh[within_deg] = _cell_i;   //
                        ++within_deg;                         //
                    } else {                                  // unmatched
                        between_dist[between_deg] = d;        //
                        between_neigh[between_deg] = _cell_i; //
                        ++between_deg;                        //
                    }
                    pq.pop();
                }
            }

            // BBKNN-inspired, Kernelized weights
            normalize_weights(within_deg, within_dist, within_weights);
            normalize_weights(between_deg, between_dist, between_weights);

            // take care of the leftover (if it exists)
            for (Index k = within_deg; k < knn_max; ++k) {
                within_dist[k] = 0;
                within_weights[k] = 0;
                if (within_deg > 0)
                    within_neigh[k] = within_neigh[within_deg - 1];
            }

            for (Index k = between_deg; k < knn_max; ++k) {
                between_dist[k] = 0;
                between_weights[k] = 0;
                if (between_deg > 0)
                    between_neigh[k] = between_neigh[between_deg - 1];
            }

            // Take the weighted average of matched data points
            {
                Mat _y0 = read_y_block(within_neigh);  // D x n
                Mat _z0 = read_z_block(within_neigh);  // K x n
                Vec w0 = eigen_vector(within_weights); // n x 1

                const Scalar denom = w0.sum(); // must be > 0

                y0_within.col(jth) = _y0 * w0 / denom;
                z0_within.col(jth) = _z0 * w0 / denom;

                const Scalar _z = z0_within.col(jth).sum(); // normalize
                z0_within.col(jth) /= _z;                   // to 1
            }
            {
                Mat _y0 = read_y_block(between_neigh);  // D x n
                Mat _z0 = read_z_block(between_neigh);  // K x n
                Vec w0 = eigen_vector(between_weights); // n x 1

                const Scalar denom = w0.sum(); // must be > 0

                y0_between.col(jth) = _y0 * w0 / denom;
                z0_between.col(jth) = _z0 * w0 / denom;

                const Scalar _z = z0_between.col(jth).sum(); // normalize
                z0_between.col(jth) /= _z;                   // to 1
            }
        }

        return std::make_tuple(y0_within, z0_within, y0_between, z0_between);
    };

    const Scalar a0 = options.gamma_a0, b0 = options.gamma_b0;

    std::vector<std::string> mu_col_names;
    mu_col_names.reserve(K * Nind);

    Mat obs_mu(D, K * Nind);
    Mat obs_mu_sd(D, K * Nind);

    Mat cf_mu(D, K * Nind);
    Mat cf_mu_sd(D, K * Nind);
    Mat cf_null_mu(D, K * Nind);
    Mat cf_null_mu_sd(D, K * Nind);

    Mat resid_mu(D, K * Nind);
    Mat resid_mu_sd(D, K * Nind);
    Mat resid_null_mu(D, K * Nind);
    Mat resid_null_mu_sd(D, K * Nind);

    const Scalar eps = 1e-4;

    for (Index ii = 0; ii < Nind; ++ii) {

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted at Ind = " << (ii));
            return EXIT_FAILURE;
        }
#endif

        Mat y0_within, z0_within;
        Mat y0_between, z0_between;

        const std::vector<Index> &cells_i = indv_index_set.at(ii);
        std::tie(y0_within, z0_within, y0_between, z0_between) =
            read_cf_block(cells_i);

        Mat y = read_y_block(cells_i); // D x N
        Mat z = read_z_block(cells_i); // K x N

        TLOG("Estimating on the sample, ind=" << ii);

        {

            poisson_t pois(y, z, y0_between, z0_between, a0, b0);
            pois.optimize();

            const Mat cf_mu_i = pois.mu_DK();
            const Mat cf_mu_sd_i = pois.mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = K * ii + k;
                cf_mu.col(s) = cf_mu_i.col(k);
                cf_mu_sd.col(s) = cf_mu_sd_i.col(k);
                const std::string c =
                    indv_id_name.at(ii) + "_" + annot_name.at(k);
                mu_col_names.emplace_back(c);
            }

            const Mat resid_mu_i = pois.residual_mu_DK();
            const Mat resid_mu_sd_i = pois.residual_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = K * ii + k;
                resid_mu.col(s) = resid_mu_i.col(k);
                resid_mu_sd.col(s) = resid_mu_sd_i.col(k);
            }
        }

        {
            poisson_t pois(y, z, y0_within, z0_within, a0, b0);
            pois.optimize();

            const Mat cf_null_mu_i = pois.mu_DK();
            const Mat cf_null_mu_sd_i = pois.mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = K * ii + k;
                cf_null_mu.col(s) = cf_null_mu_i.col(k);
                cf_null_mu_sd.col(s) = cf_null_mu_sd_i.col(k);
            }

            const Mat resid_null_mu_i = pois.residual_mu_DK();
            const Mat resid_null_mu_sd_i = pois.residual_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = K * ii + k;
                resid_null_mu.col(s) = resid_null_mu_i.col(k);
                resid_null_mu_sd.col(s) = resid_null_mu_sd_i.col(k);
            }
        }

        {
            poisson_t pois(y, z, a0, b0);
            pois.optimize();

            const Mat obs_mu_i = pois.mu_DK();
            const Mat obs_mu_sd_i = pois.mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = K * ii + k;
                obs_mu.col(s) = obs_mu_i.col(k);
                obs_mu_sd.col(s) = obs_mu_sd_i.col(k);
            }
        }
    }

    TLOG("Writing down the results ...");

    write_vector_file(options.out + ".mu_cols.gz", mu_col_names);

    write_data_file(options.out + ".cf_mu.gz", cf_mu);
    write_data_file(options.out + ".cf_mu_sd.gz", cf_mu_sd);
    write_data_file(options.out + ".obs_mu.gz", obs_mu);
    write_data_file(options.out + ".obs_mu_sd.gz", obs_mu_sd);
    write_data_file(options.out + ".resid_mu.gz", resid_mu);
    write_data_file(options.out + ".resid_mu_sd.gz", resid_mu_sd);
    write_data_file(options.out + ".resid_null_mu.gz", resid_null_mu);
    write_data_file(options.out + ".resid_null_mu_sd.gz", resid_null_mu_sd);

    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
parse_cfa_options(const int argc,     //
                  const char *argv[], //
                  OPTIONS &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)           : data MTX file (M x N)\n"
        "--data (-m)          : data MTX file (M x N)\n"
        "--col (-c)           : data column file (N x 1)\n"
        "--annot (-a)         : annotation/clustering assignment (N x 2)\n"
        "--annot_prob (-A)    : annotation/clustering probability (N x K)\n"
        "--ind (-i)           : N x 1 sample to individual (n)\n"
        "--trt_ind (-t)       : N x 1 sample to case-control membership\n"
        "--lab (-l)           : K x 1 annotation label name (e.g., cell type) \n"
        "--out (-o)           : Output file header\n"
        "\n"
        "[Options]\n"
        "\n"
        "--col_norm (-C)      : Column normalization (default: 10000)\n"
        "--normalize (-z)     : Normalize columns (default: false) \n"
        "\n"
        "--discretize (-D)    : Use discretized annotation matrix (default: true)\n"
        "--probabilistic (-P) : Use expected annotation matrix (default: false)\n"
        "\n"
        "--gamma_a0           : prior for gamma distribution(a0,b0) (default: 1)"
        "--gamma_b0           : prior for gamma distribution(a0,b0) (default: 1)"
        "\n"
        "[Matching options]\n"
        "\n"
        "--knn (-k)           : k nearest neighbours (default: 1)\n"
        "--bilink (-b)        : # of bidirectional links (default: 5)\n"
        "--nlist (-n)         : # nearest neighbor lists (default: 5)\n"
        "\n"
        "--rank (-r)          : # of SVD factors (default: rank = 50)\n"
        "--lu_iter (-u)       : # of LU iterations (default: iter = 5)\n"
        "--row_weight (-w)    : Feature re-weighting (default: none)\n"
        "\n"
        "--log_scale (-L)     : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)     : Data in a raw-scale (default: true)\n"
        "\n"
        "[Output]\n"
        "\n"
        "${out}.obs_mu.gz     : (M x n) observed matrix\n"
        "${out}.cf_mu.gz      : (M x n) confounding factors matrix\n"
        "${out}.adj_mu.gz     : (M x n) adjusted matrix\n"
        "${out}.mu_col.gz     : (n x 1) column names\n"
        "${out}.mtx.gz        : sparse (D x N) matrix\n"
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

    const char *const short_opts =
        "m:c:a:A:i:l:t:o:LRS:r:u:w:g:G:BDPC:k:b:n:hzv0:1:";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "annot_prob", required_argument, nullptr, 'A' }, //
          { "annot", required_argument, nullptr, 'a' },      //
          { "col", required_argument, nullptr, 'c' },        //
          { "ind", required_argument, nullptr, 'i' },        //
          { "trt", required_argument, nullptr, 't' },        //
          { "trt_ind", required_argument, nullptr, 't' },    //
          { "lab", required_argument, nullptr, 'l' },        //
          { "label", required_argument, nullptr, 'l' },      //
          { "out", required_argument, nullptr, 'o' },        //
          { "log_scale", no_argument, nullptr, 'L' },        //
          { "raw_scale", no_argument, nullptr, 'R' },        //
          { "block_size", required_argument, nullptr, 'S' }, //
          { "rank", required_argument, nullptr, 'r' },       //
          { "lu_iter", required_argument, nullptr, 'u' },    //
          { "row_weight", required_argument, nullptr, 'w' }, //
          { "discretize", no_argument, nullptr, 'D' },       //
          { "probabilistic", no_argument, nullptr, 'P' },    //
          { "col_norm", required_argument, nullptr, 'C' },   //
          { "knn", required_argument, nullptr, 'k' },        //
          { "bilink", required_argument, nullptr, 'b' },     //
          { "nlist", required_argument, nullptr, 'n' },      //
          { "normalize", no_argument, nullptr, 'z' },        //
          { "a0", required_argument, nullptr, '0' },         //
          { "b0", required_argument, nullptr, '1' },         //
          { "gamma_a0", required_argument, nullptr, '0' },   //
          { "gamma_a1", required_argument, nullptr, '1' },   //
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
            options.annot_name_file = std::string(optarg);
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

        case 'P':
            options.discretize = false;
            break;

        case 'C':
            options.col_norm = std::stof(optarg);
            break;

        case 'D':
            options.discretize = true;
            break;

        case 'S':
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

        case '0':
            options.gamma_a0 = std::stof(optarg);
            break;

        case '1':
            options.gamma_b0 = std::stof(optarg);
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
    ERR_RET(!file_exists(options.col_file), "No COL file");
    ERR_RET(!file_exists(options.annot_name_file), "No LAB file");

    ERR_RET(options.rank < 1, "Too small rank");

    return EXIT_SUCCESS;
}

#endif
