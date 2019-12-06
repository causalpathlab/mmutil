#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative features to reduce computational cost"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " top_features mtx_file feature_file output"
            << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "We prioritize features by their prevalence across samples (columns)."
      << std::endl;
  std::cerr << std::endl;
}

template <typename Vec>
auto eigen_argsort(const Vec& data) {
  using Index = typename Vec::Index;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data(lhs) > data(rhs); });
  return index;
}

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Scalar = float;
  using Index = long int;
  using SpMat = Eigen::SparseMatrix<Scalar>;
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

  // Calculate some scores
  const SpMat X = build_eigen_sparse(Tvec, max_row, max_col);

  using Mat = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  Vec RowScores = X * Mat::Ones(X.cols(), 1);

  auto order = eigen_argsort(RowScores);

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
  auto out_scores = std_vector(out_X * Mat::Ones(out_X.cols(), 1));

  Str output_mtx_file = output + ".mtx.gz";
  Str output_feature_file = output + ".features.gz";
  Str output_score_file = output + ".scores.gz";

  write_vector_file(output_feature_file, out_features);
  write_vector_file(output_score_file, out_scores);
  write_matrix_market_file(output_mtx_file, out_X);

  return EXIT_SUCCESS;
}
