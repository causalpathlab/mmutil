#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative features to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " top_features mtx_file feature_file output" << std::endl;
  std::cerr << std::endl;
  std::cerr << "We prioritize features by their prevalence across samples." << std::endl;
  std::cerr << std::endl;
}

using Triplet = std::tuple<Index, Index, Scalar>;
using TripletVec = std::vector<Triplet>;

struct TripletCompare {
  inline bool operator()(const Triplet& lhs, const Triplet& rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
  }
};

struct RowStatCollector {
  using index_t = Index;
  using scalar_t = Scalar;

  explicit RowStatCollector() {
    max_row = 0;
    max_col = 0;
    max_elem = 0;
    TLOG("Start reading a list of triplets");
  }

  void set_dimension(const index_t r, const index_t c, const index_t e) {
    max_row = r;
    max_col = c;
    max_elem = e;

    Row_S1.resize(max_row);
    Row_S1.setZero();
    Row_S2.resize(max_row);
    Row_S2.setZero();
    Row_N.resize(max_row);
    Row_N.setZero();
  }

  void eval(const index_t row, const index_t col, const scalar_t weight) {
    if (row < max_row && col < max_col) {
      Row_S1(row) += weight;
      Row_S2(row) += (weight * weight);
      Row_N(row)++;
    }
  }

  void eval_end() { TLOG("Finished reading a list of triplets"); }

  Index max_row;
  Index max_col;
  Index max_elem;
  Vec Row_S1;
  Vec Row_S2;
  Vec Row_N;
};

inline auto compute_mtx_stat_file(const std::string filename) {
  RowStatCollector collector;
  visit_matrix_market_file(filename, collector);

  ////////////////////////////
  // MLE standard deviation //
  ////////////////////////////

  const Vec& s1 = collector.Row_S1;
  const Vec& s2 = collector.Row_S2;
  const Scalar n = static_cast<Scalar>(collector.max_col);

  Vec ret = s2 - s1.cwiseProduct(s1 / n);
  ret = ret / std::max(n - 1.0, 1.0);
  ret = ret.cwiseSqrt();
  return ret;
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

  ////////////////////////////////
  // First calculate row scores //
  ////////////////////////////////

  Vec RowScores = compute_mtx_stat_file(mtx_file);

  /////////////////////
  // Prioritize rows //
  /////////////////////

  auto order = eigen_argsort_descending(RowScores);

  TLOG("row scores: " << RowScores(order.at(0)) << " ~ " << RowScores(order.at(order.size() - 1)));

  // TripletVec Tvec;
  // Index max_row, max_col;
  // std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);
  // std::vector<Str> features(0);
  // CHECK(read_vector_file(feature_file, features));

  // // Output
  // const Index Nout = std::min(Ntop, max_row);

  // std::vector<Str> out_features(Nout);
  // std::unordered_map<Index, Index> remap;

  // for (Index i = 0; i < Nout; ++i) {
  //   const Index j = order.at(i);
  //   out_features[i] = features.at(j);
  //   remap[j] = i;
  // }

  // TripletVec out_Tvec;

  // for (auto tt : Tvec) {
  //   Index r, c;
  //   Scalar w;
  //   std::tie(r, c, w) = tt;
  //   if (remap.count(r) > 0) {
  //     const Index new_r = remap.at(r);
  //     out_Tvec.push_back(Triplet(new_r, c, w));
  //   }
  // }

  // const SpMat out_X = build_eigen_sparse(out_Tvec, Nout, max_col);
  // auto out_scores = std_vector(row_score_sd(out_X));
  // auto out_full_scores = std_vector(RowScores);

  // Str output_mtx_file = output + ".mtx.gz";
  // Str output_feature_file = output + ".rows.gz";
  // Str output_score_file = output + ".scores.gz";
  // Str output_full_score_file = output + ".full_scores.gz";

  // write_vector_file(output_feature_file, out_features);
  // write_vector_file(output_score_file, out_scores);
  // write_vector_file(output_full_score_file, out_full_scores);
  // write_matrix_market_file(output_mtx_file, out_X);

  return EXIT_SUCCESS;
}
