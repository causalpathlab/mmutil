#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative features to reduce computational cost"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " top_features mtx_file feature_file output"
            << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "We prioritize features by their prevalence across samples."
      << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Index Ntop = boost::lexical_cast<Index>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str feature_file(argv[3]);
  const Str output(argv[4]);

  using Triplet = std::tuple<Index, Index, Scalar>;
  using TripletVec = std::vector<Triplet>;

  TripletVec Tvec;
  Index max_row, max_col;
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);

  std::vector<Str> features(0);
  CHECK(read_vector_file(feature_file, features));

  // Calculate some scores on the sparse matrix
  const SpMat X = build_eigen_sparse(Tvec, max_row, max_col);

  //////////////////////////
  // Calculate the degree //
  //////////////////////////

  auto _score_degree = [](const SpMat& xx) {
    return xx.unaryExpr([](const Scalar x) { return std::abs(x); }) *
           Mat::Ones(xx.cols(), 1);
  };

  auto _score_sd = [](const SpMat& xx) {
    Vec s1 = xx * Mat::Ones(xx.cols(), 1);
    Vec s2 = xx.cwiseProduct(xx) * Mat::Ones(xx.cols(), 1);
    const Scalar n = xx.cols();
    Vec ret = s2 - s1.cwiseProduct(s1 / n);
    ret = ret / std::max(n - 1.0, 1.0);
    ret = ret.cwiseSqrt();
    return ret;
  };

  /////////////////////
  // Prioritize rows //
  /////////////////////

  Vec RowScores = _score_sd(X);

  auto order = eigen_argsort_descending(RowScores);

  TLOG("row scores: " << RowScores(order.at(0)) << " ~ "
                      << RowScores(order.at(order.size() - 1)));

  // Output
  const Index Nout = std::min(Ntop, max_row);

  std::vector<Str> out_features(Nout);
  std::unordered_map<Index, Index> remap;

  for (Index i = 0; i < Nout; ++i) {
    const Index j = order.at(i);
    out_features[i] = features.at(j);
    remap[j] = i;
  }

  TripletVec out_Tvec;

  for (auto tt : Tvec) {
    Index r, c;
    Scalar w;
    std::tie(r, c, w) = tt;
    if (remap.count(r) > 0) {
      const Index new_r = remap.at(r);
      out_Tvec.push_back(Triplet(new_r, c, w));
    }
  }

  const SpMat out_X = build_eigen_sparse(out_Tvec, Nout, max_col);
  auto out_scores = std_vector(_score_sd(out_X));

  Str output_mtx_file = output + ".mtx.gz";
  Str output_feature_file = output + ".features.gz";
  Str output_score_file = output + ".scores.gz";

  write_vector_file(output_feature_file, out_features);
  write_vector_file(output_score_file, out_scores);
  write_matrix_market_file(output_mtx_file, out_X);

  return EXIT_SUCCESS;
}
