    const std::string mtx_file = options.mtx_file;
    const std::string idx_file = options.mtx_file + ".index";
    const std::string annot_prob_file = options.annot_prob_file;
    const std::string annot_file = options.annot_file;
    const std::string col_file = options.col_file;
    const std::string ind_file = options.ind_file;
    const std::string trt_file = options.trt_ind_file;
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

    ASSERT(indv_membership.size() == Nsample,
           "Check the individual membership file: "
               << indv_membership.size() << " vs. expected N = " << Nsample);

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

    std::vector<std::string> trt_id_name; //
    std::vector<Index> trt_map;           // map: col -> trt index

    CHECK(read_vector_file(trt_file, trt_membership));

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
        } else {
            zz = zz_prob;
        }
        return zz;
    };

    /////////////////////
    // add data points //
    /////////////////////

    const Index rank = proj.cols();

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

    using vs_type = hnswlib::InnerProductSpace;

    ///////////////////////////////////////////////////////
    // construct dictionary for each treatment condition //
    ///////////////////////////////////////////////////////

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec_trt;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_trt;

    for (Index tt = 0; tt < Ntrt; ++tt) {

        const Index n_tot = trt_index_set[tt].size();

        vs_vec_trt.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_trt[tt].get();

        knn_lookup_trt.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

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

    const Index glm_reg = options.glm_reg;
    const Index glm_iter = options.glm_iter;
    const Scalar glm_pseudo = options.glm_pseudo;

    auto glm_feature = [&glm_pseudo](const Scalar &x) -> Scalar {
        return fasterlog(x + glm_pseudo);
    };

    /// Construct counterfactually-matched blocks
    /// @param ind_i individual i [0, Nind)
    /// @returns (y0_internal, y0_counterfactual)
    /// D x n_i and K x n_i
    auto construct_cf_block = [&](const std::vector<Index> &cells_j) {
        float *mass = V.data();
        const Index n_j = cells_j.size();

        Mat y = read_y_block(cells_j);
        Mat y0_internal(D, n_j);
        Mat y0_counterfactual(D, n_j);
        y0_internal.setZero();
        y0_counterfactual.setZero();

#pragma omp parallel for
        for (Index jth = 0; jth < n_j; ++jth) {    // For each cell j
            const Index _cell_j = cells_j.at(jth); //
            const Index tj = trt_map.at(_cell_j);  // Trt group for this cell j
            const Index jj = indv.at(_cell_j);     // Individual for this cell j
            const std::size_t n_j = cells_j.size(); // number of cells

            // Index internal_deg = 0;       // # of neighbours
            // Index counterfactual_deg = 0; // # of neighbours
            // std::vector<Scalar> internal_dist;
            // std::vector<Scalar> counterfactual_dist;
            std::vector<Index> internal_neigh;
            std::vector<Index> counterfactual_neigh;

#ifdef CPYTHON
            if (PyErr_CheckSignals() != 0) {
                ELOG("Interrupted while working on kNN: j = " << jth);
                std::exit(1);
            }
#endif

            for (Index ii = 0; ii < Nind; ++ii) {

                const std::vector<Index> &cells_i = indv_index_set.at(ii);
                KnnAlg &alg_ii = *knn_lookup_indv[ii].get();
                const std::size_t n_i = cells_i.size();
                const std::size_t nquery = std::min(knn_each, n_i);

                auto pq =
                    alg_ii.searchKnn((void *)(mass + rank * _cell_j), nquery);

                while (!pq.empty()) {
                    float d = 0;                          // distance
                    std::size_t k;                        // local index
                    std::tie(d, k) = pq.top();            //
                    const Index _cell_i = cells_i.at(k);  // global index
                    const Index ti = trt_map.at(_cell_i); // treatment

                    if (_cell_j == _cell_i) { // Skip the same cell
                        pq.pop();             // Let's move on
                        continue;             //
                    }

                    // internal
                    if (ti == tj) {
                        // internal_dist.emplace_back(d);
                        internal_neigh.emplace_back(_cell_i);
                        // ++internal_deg;
                    }

                    if (ti != tj) {
                        // counterfactual_dist.emplace_back(d);
                        counterfactual_neigh.emplace_back(_cell_i);
                        // ++counterfactual_deg;
                    }
                    pq.pop();
                }
            }

            // std::vector<Scalar> counterfactual_weights(counterfactual_deg);
            // std::vector<Scalar> internal_weights(internal_deg);

            ////////////////////////////////////////////////////////
            // Find optimal weights for counterfactual imputation //
            ////////////////////////////////////////////////////////

            Mat yy = y.col(jth);

            {
                Mat xx =
                    read_y_block(counterfactual_neigh).unaryExpr(glm_feature);

                y0_counterfactual.col(jth) =
                    predict_poisson_glm(xx, yy, glm_iter, glm_reg);
            }

            {
                Mat xx = read_y_block(internal_neigh).unaryExpr(glm_feature);

                y0_internal.col(jth) =
                    predict_poisson_glm(xx, yy, glm_iter, glm_reg);
            }
        }

        return std::make_tuple(y0_internal, y0_counterfactual);
    };
