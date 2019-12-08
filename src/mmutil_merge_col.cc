#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Merge the columns of sparse matrices matching rows" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " master_feature count_threshold output" << std::endl;
  std::cerr << " { mtx[1] feature[1] mtx[2] feature[2] ... }" << std::endl;
  std::cerr << std::endl;
  std::cerr << "master_feature  : A file that contains the names of features."
            << std::endl;
  std::cerr << "output          : Header string for the output fileset."
            << std::endl;
  std::cerr << "count_threshold : Set the minimum sum of features per column"
            << std::endl;
  std::cerr << "mtx[i]          : i-th matrix market format file" << std::endl;
  std::cerr << "feature[i]      : i-th feature file" << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 6) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;
  using Str2Index = std::unordered_map<Str, Index>;
  using Index2Str = std::vector<Str>;
  using Index2Index = std::vector<Index>;

  const Index NA = -1;
  using Triplet = std::tuple<Index, Index, Scalar>;
  using TripletVec = std::vector<Triplet>;
  using _Triplet = Eigen::Triplet<Scalar>;
  using _TripletVec = std::vector<_Triplet>;

  ///////////////////////////////////
  // first read universal features //
  ///////////////////////////////////

  const Str univ_feature_file(argv[1]);
  const Scalar column_threshold = boost::lexical_cast<Scalar>(argv[2]);
  const Str output(argv[3]);

  Index2Str univ_features(0);
  CHECK(read_vector_file(univ_feature_file, univ_features));
  TLOG("Read the universal feature names: "
       << univ_feature_file << " " << univ_features.size() << " features");

  Str2Index univ_positions;
  const Index univ_max_row = univ_features.size();

  for (Index r = 0; r < univ_features.size(); ++r) {
    univ_positions[univ_features.at(r)] = r;
  }

  // Do the basic Q/C for each pair of files

  Index univ_max_col = 0;
  _TripletVec univ_Tvec;
  using IndexPair = std::pair<Index, Index>;  // batch, column
  std::vector<IndexPair> univ_columns;
  Index batch_index = 0;

  for (int j = 4; j < argc; j += 2) {
    std::string mtx_file(argv[j]);
    std::string feature_file(argv[j + 1]);
    std::vector<std::string> features(0);
    CHECK(read_vector_file(feature_file, features));

    TLOG("Processing : " << mtx_file << ", " << feature_file);

    // a. Learn how to repmap them
    Index2Index remap(features.size(), NA);
    for (Index _from = 0; _from < features.size(); ++_from) {
      const auto& s = features.at(_from);
      if (univ_positions.count(s) > 0) {
        Index _to = univ_positions.at(s);
        remap[_from] = _to;
      }
    }

    // b. Read all the triplets
    TripletVec Tvec_all;
    Index max_row, max_col;
    std::tie(Tvec_all, max_row, max_col) = read_matrix_market_file(mtx_file);

    TLOG("The list of all triplets : " << Tvec_all.size());

    // c. Select those containing relevant features
    _TripletVec Tvec_trim;

    for (auto tt : Tvec_all) {
      Index r;
      Index c;
      Scalar w;
      std::tie(r, c, w) = tt;

      if (remap.at(r) != NA) {
        const auto new_r = remap.at(r);
        Tvec_trim.push_back(_Triplet(new_r, c, w));
      }
    }
    TLOG("Trimmed the list of triplets : " << Tvec_trim.size());

    SpMat X0(univ_max_row, max_col);
    X0.setFromTriplets(Tvec_trim.begin(), Tvec_trim.end());
    TLOG("Constructed the sparse matrix : " << X0.rows() << " x " << X0.cols());

    // d. Trim the columns
    SpMat X;
    std::vector<Index> Columns;
    std::tie(X, Columns) = filter_columns(X0, column_threshold);

    TLOG("Eliminated the under-sampled : " << X.rows() << " x " << X.cols());

    // e. Save
    for (Index j = 0; j < X.outerSize(); ++j) {
      for (SpMat::InnerIterator it(X, j); it; ++it) {
        const Index row = it.row();
        const Index col = j + univ_max_col;
        const Scalar val = it.value();
        univ_Tvec.push_back(_Triplet(row, col, val));
      }
    }

    for (auto j : Columns) {
      univ_columns.push_back(IndexPair(batch_index + 1, j + 1));
    }

    batch_index++;
    univ_max_col += X.cols();
  }

  // Construct final sparse matrix
  SpMat X(univ_max_row, univ_max_col);
  X.setFromTriplets(univ_Tvec.begin(), univ_Tvec.end());

  Str output_mtx = output + ".mtx.gz";
  Str output_column = output + ".columns.gz";

  write_pair_file(output_column, univ_columns);
  write_matrix_market_file(output_mtx, X);

  return EXIT_SUCCESS;
}
