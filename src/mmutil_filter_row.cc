#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative features to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " top_features mtx_file feature_file output" << std::endl;
  std::cerr << std::endl;
  std::cerr << "We prioritize features by their prevalence across samples." << std::endl;
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

  /////////////////////////////
  // Sort the triplet vector //
  /////////////////////////////

  TLOG("Sorting the triplet vector by the row index...");

  std::sort(Tvec.begin(), Tvec.end(), [](const Triplet& lhs, const Triplet& rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
  });

  Vec RowScores(max_row);
  const Index BATCH_SIZE = 1000;

  TLOG("Calculating the scores...");

  Index _tvec_start = 0;

  // Do the calculation for every 1000 features, [start, end)
  for (Index _start_row = 0; _start_row < max_row; _start_row += BATCH_SIZE) {
    const Index _end_row = std::min(_start_row + BATCH_SIZE, max_row);
    TripletVec _tvec;

    // Identify [_tvec_start, _tvec_end)
    Index _tvec_end = _tvec_start;
    while (_tvec_end < Tvec.size()) {
      if (std::get<0>(Tvec.at(_tvec_end)) >= _end_row) {
        break;
      }
      _tvec.push_back(Tvec.at(_tvec_end));
      ++_tvec_end;
    }

#ifdef DEBUG
    for (auto tt : _tvec) {
      ASSERT((std::get<0>(tt) >= _start_row && std::get<0>(tt) < _end_row),
             "Check: " << _start_row << " ~ " << _end_row << "\n"
                       << "Check: " << _tvec_start << " ~ " << _tvec_end);
    }
#endif

    // Construct a cumulative, yet sparser, matrix
    const SpMat X = build_eigen_sparse(_tvec, _end_row, max_col);
    Vec _row_scores = row_score_sd(X);
    for (Index r = _start_row; r < _row_scores.size(); ++r) {
      RowScores(r) = _row_scores(r);
    }
    // TLOG("Done [" << _start_row << ", " << _end_row << ")");
    _tvec_start = _tvec_end;
  }

  /////////////////////
  // Prioritize rows //
  /////////////////////

  auto order = eigen_argsort_descending(RowScores);

  TLOG("row scores: " << RowScores(order.at(0)) << " ~ " << RowScores(order.at(order.size() - 1)));

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
  auto out_scores = std_vector(row_score_sd(out_X));
  auto out_full_scores = std_vector(RowScores);

  Str output_mtx_file = output + ".mtx.gz";
  Str output_feature_file = output + ".rows.gz";
  Str output_score_file = output + ".scores.gz";
  Str output_full_score_file = output + ".full_scores.gz";

  write_vector_file(output_feature_file, out_features);
  write_vector_file(output_score_file, out_scores);
  write_vector_file(output_full_score_file, out_full_scores);
  write_matrix_market_file(output_mtx_file, out_X);

  return EXIT_SUCCESS;
}
