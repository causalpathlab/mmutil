#include "mmutil.hh"
#include "mmutil_spectral.hh"

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
  TripletVec univ_Tvec;
  using IndexPair = std::pair<Str, Index>;  // column, batch
  std::vector<IndexPair> univ_column_names;
  Index batch_index = 0;

  for (int b = 4; b < argc; b += 3) {
    const Str mtx_file(argv[b]);
    const Str row_file(argv[b + 1]);
    const Str col_file(argv[b + 2]);

    TLOG("MTX : " << mtx_file);
    TLOG("ROW : " << row_file);
    TLOG("COL : " << col_file);

    std::vector<Str> row_names(0);
    CHECK(read_vector_file(row_file, row_names));
    std::vector<Str> column_names(0);
    CHECK(read_vector_file(col_file, column_names));

    // a. Learn how to repmap them
    Index2Index remap(row_names.size(), NA);
    for (Index _from = 0; _from < row_names.size(); ++_from) {
      const Str& s = row_names.at(_from);
      if (univ_positions.count(s) > 0) {
        const Index _to = univ_positions.at(s);
        remap[_from] = _to;
      }
    }

    // b. Read all the triplets
    TripletVec Tvec_all;
    Index max_row, max_col;
    std::tie(Tvec_all, max_row, max_col) = read_matrix_market_file(mtx_file);

    TLOG("The list of all triplets --> " << Tvec_all.size());

    ASSERT(max_col == column_names.size(), "each column needs a name");

    // c. Select those containing relevant row_names
    _TripletVec Tvec_relevant;

    for (auto tt : Tvec_all) {
      Index r;
      Index c;
      Scalar w;
      std::tie(r, c, w) = tt;

      if (remap.at(r) != NA) {
        const Index new_r = remap.at(r);
        Tvec_relevant.push_back(_Triplet(new_r, c, w));
      }
    }

    TLOG("Trimmed the list of triplets --> " << Tvec_relevant.size());

    SpMat X0(univ_max_row, max_col);
    X0.reserve(Tvec_relevant.size());
    X0.setFromTriplets(Tvec_relevant.begin(), Tvec_relevant.end());
    TLOG("Built the sparse matrix --> " << X0.rows() << " x " << X0.cols());

    // d. Trim the columns
    SpMat X;
    std::vector<Index> trimmed_columns;
    std::tie(X, trimmed_columns) = filter_columns(X0, column_threshold);

    TLOG("After removing low-quality columns -> " << X.rows() << " x " << X.cols());

#ifdef DEBUG
    Index _max_row = 0;
    Index _max_col = 0;
#endif

    // e. Save
    for (Index j = 0; j < X.outerSize(); ++j) {
      for (SpMat::InnerIterator it(X, j); it; ++it) {
        const Index _row = it.row();
        const Index _col = it.col() + univ_max_col;
        const Scalar _val = it.value();

#ifdef DEBUG
        if (_row > _max_row) _max_row = _row;
        if (_col > _max_col) _max_col = _col;
#endif
        univ_Tvec.push_back(Triplet(_row, _col, _val));
      }
    }

    TLOG("Keep " << univ_Tvec.size() << " triplets");

    for (Index j : trimmed_columns) {
      univ_column_names.push_back(IndexPair(column_names.at(j), batch_index + 1));
    }

    batch_index++;
    univ_max_col += X.cols();

#ifdef DEBUG
    ASSERT(_max_row < univ_max_row, "invalid row");
    ASSERT(_max_col < univ_max_col, "invalid col");
#endif
  }

  TLOG("Imported all the data files");

  Str output_mtx = output + ".mtx.gz";
  Str output_column = output + ".columns.gz";

  TLOG("Output the columns --> " << output_column);
  write_pair_file(output_column, univ_column_names);

  TLOG("Number of non-zero elements = " << univ_Tvec.size());
  {
    ogzstream ofs(output_mtx.c_str(), std::ios::out);

    ofs.precision(4);

    ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
    ofs << univ_max_row << " ";
    ofs << univ_max_col << " ";
    ofs << univ_Tvec.size() << std::endl;

    ///////////////////////////////////////////////
    // note: matrix market uses 1-based indexing //
    ///////////////////////////////////////////////

    Index i, j;
    Scalar v;

    const Index INTERVAL = 1e6;
    const Index max_triples = univ_Tvec.size();
    const Index MAX_PRINT = (max_triples / INTERVAL);
    Index _num_triples = 0;

    for (auto tt : univ_Tvec) {
      std::tie(i, j, v) = tt;
      ofs << (i + 1) << " " << (j + 1) << " " << v << std::endl;

      if (++_num_triples % INTERVAL == 0) {
        const Index _wrote = (_num_triples / INTERVAL);
        std::cerr << "\r";
        std::cerr << std::left << std::setfill('.') << std::setw(30) << "Writing ";
        std::cerr << std::setfill(' ') << std::setw(10) << _wrote;
        std::cerr << " x 1M (total " << std::setw(10) << MAX_PRINT << ")\r";
        std::cerr << std::flush;
      }
    }

    ofs.close();
  }
  TLOG("Wrote " << output_mtx << " file");

  Str output_row = output + ".rows.gz";
  write_vector_file(output_row, univ_rows);

  TLOG("Wrote " << output_row << " (for the record)");

  TLOG("Successfully finished");
  return EXIT_SUCCESS;
}
