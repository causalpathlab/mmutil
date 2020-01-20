#include "mmutil_match.hh"

int
main(const int argc, const char* argv[]) {

  match_options_t mopt;

  CHECK(parse_match_options(argc, argv, mopt));

  const std::string mtx_src_file(mopt.src_mtx);
  const std::string mtx_tgt_file(mopt.tgt_mtx);

  ERR_RET(!file_exists(mtx_src_file), "No source data file");
  ERR_RET(!file_exists(mtx_tgt_file), "No target data file");

  const std::string out_file(mopt.out);

  std::vector<std::tuple<Index, Index, Scalar> > out_index;

  const SpMat Src = build_eigen_sparse(mtx_src_file).transpose().eval();
  const SpMat Tgt = build_eigen_sparse(mtx_tgt_file).transpose().eval();

  auto knn = search_knn(SrcSparseRowsT(Src),  //
                        TgtSparseRowsT(Tgt),  //
                        KNN(mopt.knn),        //
                        BILINK(mopt.bilink),  //
                        NNLIST(mopt.nlist),   //
                        out_index);

  CHK_ERR_RET(knn, "Failed to search kNN");

  /////////////////////////////
  // fliter out zero columns //
  /////////////////////////////

  auto valid_src = find_nz_cols(mtx_src_file);
  auto valid_tgt = find_nz_cols(mtx_tgt_file);

  TLOG("Filter out total zero columns");

  ///////////////////////////////
  // give names to the columns //
  ///////////////////////////////

  const std::string col_src_file(mopt.src_col);
  const std::string col_tgt_file(mopt.tgt_col);

  std::vector<std::string> col_src_names;
  std::vector<std::string> col_tgt_names;

  CHECK(read_vector_file(col_src_file, col_src_names));
  CHECK(read_vector_file(col_tgt_file, col_tgt_names));

  std::vector<std::tuple<std::string, std::string, Scalar> > out_named;

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
