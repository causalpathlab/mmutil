#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Merge the columns of sparse matrices matching rows" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " master_row count_threshold output" << std::endl;
  std::cerr << " { mtx[1] row[1] column[1] mtx[2] row[2] column[2] ... }" << std::endl;
  std::cerr << std::endl;
  std::cerr << "master_row  : A file that contains the names of rows." << std::endl;
  std::cerr << "count_threshold : Set the minimum sum of rows per column" << std::endl;
  std::cerr << "output          : Header string for the output fileset." << std::endl;
  std::cerr << "mtx[i]          : i-th matrix market format file" << std::endl;
  std::cerr << "row[i]          : i-th row file" << std::endl;
  std::cerr << "column[i]       : i-th column file" << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 7) {
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
  // first read universal rows //
  ///////////////////////////////////

  const Str univ_row_file(argv[1]);
  const Scalar column_threshold = boost::lexical_cast<Scalar>(argv[2]);
  const Str output(argv[3]);

  Index2Str univ_rows(0);
  CHECK(read_vector_file(univ_row_file, univ_rows));
  TLOG("Read the universal row names: " << univ_row_file << " " << univ_rows.size() << " rows");

  Str2Index univ_positions;
  const Index univ_max_row = univ_rows.size();

  for (Index r = 0; r < univ_rows.size(); ++r) {
    univ_positions[univ_rows.at(r)] = r;
  }

  // Do the basic Q/C for each pair of files

  Index univ_max_col = 0;
  _TripletVec univ_Tvec;
  using IndexPair = std::pair<Str, Index>;  // column, batch
  std::vector<IndexPair> univ_columns;
  Index batch_index = 0;

  for (int b = 4; b < argc; b += 3) {
    const std::string mtx_file(argv[b]);
    const std::string row_file(argv[b + 1]);
    const std::string col_file(argv[b + 2]);

    TLOG("MTX : " << mtx_file);
    TLOG("ROW : " << row_file);
    TLOG("COL : " << col_file);

    std::vector<std::string> row_names(0);
    CHECK(read_vector_file(row_file, row_names));
    std::vector<std::string> column_names(0);
    CHECK(read_vector_file(col_file, column_names));

    // a. Learn how to repmap them
    Index2Index remap(row_names.size(), NA);
    for (Index _from = 0; _from < row_names.size(); ++_from) {
      const auto& s = row_names.at(_from);
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

    // c. Select those containing relevant row_names
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

    TLOG("Eliminated the under-sampled -> " << X.rows() << " x " << X.cols());

    // e. Save
    for (Index j = 0; j < X.outerSize(); ++j) {
      for (SpMat::InnerIterator it(X, j); it; ++it) {
        const Index _row = it.row();
        const Index _col = it.col() + univ_max_col;
        const Scalar _val = it.value();
        univ_Tvec.push_back(_Triplet(_row, _col, _val));
      }
    }

    for (auto j : Columns) {
      univ_columns.push_back(IndexPair(column_names.at(j), batch_index + 1));
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
