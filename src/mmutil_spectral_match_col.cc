#include "mmutil_match.hh"
#include "mmutil_spectral.hh"
#include "mmutil_util.hh"

int
main(const int argc, const char *argv[])
{
    match_options_t options;

    CHECK(parse_match_options(argc, argv, options));

    ERR_RET(!file_exists(options.src_mtx), "No source data file");
    ERR_RET(!file_exists(options.tgt_mtx), "No target data file");

    CHECK(mmutil::bgzf::convert_bgzip(options.src_mtx));
    CHECK(mmutil::bgzf::convert_bgzip(options.tgt_mtx));
    CHECK(mmutil::index::build_mmutil_index(options.src_mtx));
    CHECK(mmutil::index::build_mmutil_index(options.tgt_mtx));

    /////////////////////////////
    // fliter out zero columns //
    /////////////////////////////

    using valid_set_t = std::unordered_set<Index>;
    using str_vec_t = std::vector<std::string>;
    str_vec_t src_names, tgt_names;
    valid_set_t valid_src, valid_tgt;
    Index Nsrc, Ntgt;

    std::tie(valid_src, Nsrc, src_names) =
        find_nz_col_names(options.src_mtx, options.src_col);
    std::tie(valid_tgt, Ntgt, tgt_names) =
        find_nz_col_names(options.tgt_mtx, options.tgt_col);

    TLOG("Filter out total zero columns");

    Vec weights;
    if (file_exists(options.row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(options.row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    ///////////////////////////////////////////////
    // step 1. learn spectral on the target data //
    ///////////////////////////////////////////////

    svd_out_t svd = take_svd_online(options.tgt_mtx, weights, options);

    Mat dict = svd.U; // feature x factor
    Mat d = svd.D;    // singular values
    Mat tgt = svd.V;  // sample x factor

    TLOG("Target matrix: " << tgt.rows() << " x " << tgt.cols());

    /////////////////////////////////////////////////////
    // step 2. project source data onto the same space //
    /////////////////////////////////////////////////////

    Mat proj = dict * d.cwiseInverse().asDiagonal(); // feature x rank

    Mat src = take_proj_online(options.src_mtx, weights, proj, options);

    TLOG("Source matrix: " << src.rows() << " x " << src.cols());

    //////////////////////////////
    // step 3. search kNN pairs //
    //////////////////////////////

    ERR_RET(src.cols() != tgt.cols(),
            "Found different number of spectral features:"
                << src.cols() << " vs. " << tgt.cols());

    src.transposeInPlace(); // rank x samples
    tgt.transposeInPlace(); // rank x samples
    normalize_columns(src); // For cosine distance
    normalize_columns(tgt); //

    std::vector<std::tuple<Index, Index, Scalar>> out_index;

    TLOG("Running kNN search ...");

    auto knn = search_knn(SrcDataT(src.data(), src.rows(), src.cols()),
                          TgtDataT(tgt.data(), tgt.rows(), tgt.cols()),
                          KNN(options.knn),       //
                          BILINK(options.bilink), //
                          NNLIST(options.nlist),  //
                          out_index);

    CHK_ERR_RET(knn, "Failed to search kNN");

    auto dist2sim = [](std::tuple<Index, Index, Scalar> &tup) {
        std::get<2>(tup) = fasterexp(-std::get<2>(tup));
    };
    std::for_each(out_index.begin(), out_index.end(), dist2sim);

    TLOG("Convert distance to similarity");

    const Index max_row = src.cols();
    const Index max_col = tgt.cols();
    const SpMat A = build_eigen_sparse(out_index, max_row, max_col);

    write_matrix_market_file(options.out + ".mtx.gz", A);
    write_vector_file(options.out + ".rows.gz", src_names);
    write_vector_file(options.out + ".cols.gz", tgt_names);

    TLOG("Done");
    return EXIT_SUCCESS;
}
