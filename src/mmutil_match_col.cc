#include "mmutil_match.hh"

int
main(const int argc, const char *argv[])
{
    match_options_t options;

    CHECK(parse_match_options(argc, argv, options));

    ERR_RET(!file_exists(options.src_mtx), "No source data file");
    ERR_RET(!file_exists(options.tgt_mtx), "No target data file");

    /////////////////////////////
    // fliter out zero columns //
    /////////////////////////////

    using valid_set_t = std::unordered_set<Index>;
    using str_vec_t = std::vector<std::string>;
    str_vec_t col_src_names, col_tgt_names;
    valid_set_t valid_src, valid_tgt;
    Index Nsrc, Ntgt;

    std::tie(valid_src, Nsrc, col_src_names) =
        find_nz_col_names(options.src_mtx, options.src_col);
    std::tie(valid_tgt, Ntgt, col_tgt_names) =
        find_nz_col_names(options.tgt_mtx, options.tgt_col);

    TLOG("Filter out total zero columns");

    const std::string out_file(options.out);

    std::vector<std::tuple<Index, Index, Scalar>> out_index;

    const SpMat Src = read_eigen_sparse(options.src_mtx).transpose().eval();
    const SpMat Tgt = read_eigen_sparse(options.tgt_mtx).transpose().eval();

    auto knn = search_knn(SrcSparseRowsT(Src),    //
                          TgtSparseRowsT(Tgt),    //
                          KNN(options.knn),       //
                          BILINK(options.bilink), //
                          NNLIST(options.nlist),  //
                          out_index);

    CHK_ERR_RET(knn, "Failed to search kNN");

    std::vector<std::tuple<std::string, std::string, Scalar>> out_named;

    for (auto tt : out_index) {
        Index i, j;
        Scalar d;
        std::tie(i, j, d) = tt;
        if (valid_src.count(i) > 0 && valid_tgt.count(j) > 0) {
            out_named.push_back(
                std::make_tuple(col_src_names.at(i), col_tgt_names.at(j), d));
        }
    }

    write_tuple_file(out_file, out_named);

    TLOG("Wrote the matching file: " << out_file);

    return EXIT_SUCCESS;
}
