#include "mmutil_match.hh"
#include "mmutil_spectral.hh"

int
main(const int argc, const char* argv[]) {

  match_options_t options;

  CHECK(parse_match_options(argc, argv, options));

  ERR_RET(!file_exists(options.src_mtx), "No source data file");
  ERR_RET(!file_exists(options.tgt_mtx), "No target data file");

  /////////////////////////////
  // fliter out zero columns //
  /////////////////////////////

  using valid_set_t = std::unordered_set<Index>;
  using str_vec_t   = std::vector<std::string>;
  str_vec_t col_src_names, col_tgt_names;
  valid_set_t valid_src, valid_tgt;
  Index Nsrc, Ntgt;

  std::tie(valid_src, Nsrc, col_src_names) =
      find_nz_col_names(options.src_mtx, options.src_col);
  std::tie(valid_tgt, Ntgt, col_tgt_names) =
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

  Mat u, v, d;
  std::tie(u, v, d) = take_spectrum_nystrom(options.tgt_mtx, weights, options);

  TLOG("Target matrix: " << u.rows() << " x " << u.cols());

  /////////////////////////////////////////////////////
  // step 2. project source data onto the same space //
  /////////////////////////////////////////////////////

  Mat proj = v * d.cwiseInverse().asDiagonal();  // feature x rank

  Mat u_src = _nystrom_proj(options.src_mtx, weights, proj, options);

  TLOG("Source matrix: " << u_src.rows() << " x " << u_src.cols());

  //////////////////////////////
  // step 3. search kNN pairs //
  //////////////////////////////

  ERR_RET(u_src.cols() != u.cols(), "Found different number of features: "
                                        << u_src.cols() << " vs. " << u.cols());

  u_src.transposeInPlace();  // Column-major
  u.transposeInPlace();      //

  u_src.colwise().normalize();  // Normalize for cosine distance
  u.colwise().normalize();      //

  std::vector<std::tuple<Index, Index, Scalar> > out_index;

  TLOG("Running kNN search ...");

  auto knn = search_knn(SrcDataT(u_src.data(), u_src.rows(), u_src.cols()),
                        TgtDataT(u.data(), u.rows(), u.cols()),
                        KNN(options.knn),        //
                        BILINK(options.bilink),  //
                        NNLIST(options.nlist),   //
                        out_index);

  CHK_ERR_RET(knn, "Failed to search kNN");

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

  const std::string out_file(options.out);

  write_tuple_file(out_file, out_named);

  TLOG("Wrote the matching file: " << out_file);

  return EXIT_SUCCESS;
}
