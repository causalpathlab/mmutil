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

struct col_counter_on_valid_rows_t {
  using index_t     = Index;
  using scalar_t    = Scalar;
  using index_map_t = std::unordered_map<index_t, index_t>;

  explicit col_counter_on_valid_rows_t(const index_map_t& _valid_rows) : valid_rows(_valid_rows) {
    max_row  = 0;
    max_col  = 0;
    max_elem = 0;
  }

  void set_dimension(const index_t r, const index_t c, const index_t e) {
    std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
    Col_N.resize(max_col);
    Col_N.setZero();
  }

  void eval(const index_t row, const index_t col, const scalar_t weight) {
    if (row < max_row && col < max_col && is_valid(row)) {
      Col_N(col)++;
    }
  }

  void eval_end() {
    // TLOG("Found " << Col_N.sum() << std::endl);
  }

  const index_map_t& valid_rows;

  Index max_row;
  Index max_col;
  Index max_elem;

  Vec Col_N;

  inline bool is_valid(const index_t row) { return valid_rows.count(row) > 0; }
};

struct triplet_copier_t {

  using index_t     = Index;
  using scalar_t    = Scalar;
  using index_map_t = std::unordered_map<index_t, index_t>;

  explicit triplet_copier_t(const std::string _filename,   // output file name
			    const index_map_t& _remap_row, // row mapper
                            const index_map_t& _remap_col) // column mapper
      : filename(_filename), remap_row(_remap_row), remap_col(_remap_col) {
    max_row  = 0;
    max_col  = 0;
    max_elem = 0;
    ASSERT(remap_row.size() > 0, "Empty Remap");
    ASSERT(remap_col.size() > 0, "Empty Remap");
  }

  void set_dimension(const index_t r, const index_t c, const index_t e) {
    // just bypass this.. since we read & write multiple times
  }

  void set_global_dimension(const index_t r, const index_t c, const index_t e) {
    std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

    ofs.open(filename.c_str(), std::ios::out);
    ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
    ofs << max_row << FS << max_col << FS << max_elem << std::endl;
  }

  void eval(const index_t row, const index_t col, const scalar_t weight) {
    if (remap_col.count(col) > 0 && remap_row.count(row) > 0) {
      const index_t i = remap_row.at(row) + 1;  // fix zero-based to one-based
      const index_t j = remap_col.at(col) + 1;  // fix zero-based to one-based
      ofs << i << FS << j << FS << weight << std::endl;
    }
  }

  void eval_end() {
    // just bypass this..
  }

  void eval_global_end() { ofs.close(); }

  const std::string filename;
  static constexpr char FS = ' ';

  const index_map_t& remap_row;
  const index_map_t& remap_col;

 private:
  Index max_row;
  Index max_col;
  Index max_elem;

  ogzstream ofs;
};

//////////
// main //
//////////

int main(const int argc, const char* argv[]) {
  if (argc < 7) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str         = std::string;
  using Str2Index   = std::unordered_map<Str, Index>;
  using Index2Str   = std::vector<Str>;
  using Index2Index = std::vector<Index>;

  const Index NA    = -1;
  using Triplet     = std::tuple<Index, Index, Scalar>;
  using TripletVec  = std::vector<Triplet>;
  using _Triplet    = Eigen::Triplet<Scalar>;
  using _TripletVec = std::vector<_Triplet>;

  ////////////////////////////
  // first read global rows //
  ////////////////////////////

  const Str glob_row_file(argv[1]);
  const Scalar column_threshold = boost::lexical_cast<Scalar>(argv[2]);
  const Str output(argv[3]);

  Index2Str glob_rows(0);
  CHECK(read_vector_file(glob_row_file, glob_rows));
  TLOG("Read the global row names: " << glob_row_file << " " << glob_rows.size() << " rows");

  Str2Index glob_positions;
  const Index glob_max_row = glob_rows.size();

  for (Index r = 0; r < glob_rows.size(); ++r) {
    glob_positions[glob_rows.at(r)] = r;
  }

  auto is_glob_row   = [&glob_positions](const Str& s) { return glob_positions.count(s) > 0; };
  auto _glob_row_pos = [&glob_positions](const Str& s) { return glob_positions.at(s); };

  ///////////////////////////////
  // Figure out dimensionality //
  ///////////////////////////////

  using index_map_t     = col_counter_on_valid_rows_t::index_map_t;
  using index_pair_t    = std::pair<Index, Index>;
  using index_map_ptr_t = std::shared_ptr<index_map_t>;

  std::vector<index_map_ptr_t> remap_to_glob_row_vec;
  std::vector<index_map_ptr_t> remap_to_glob_col_vec;
  std::vector<index_map_ptr_t> remap_to_local_col_vec;

  Index glob_max_col = 0;

  for (int b = 4; b < argc; b += 3) {
    const Str mtx_file(argv[b]);
    const Str row_file(argv[b + 1]);
    const Str col_file(argv[b + 2]);

    TLOG("MTX : " << mtx_file);
    TLOG("ROW : " << row_file);
    TLOG("COL : " << col_file);

    ////////////////////////////////
    // What are overlapping rows? //
    ////////////////////////////////

    remap_to_glob_row_vec.push_back(std::make_shared<index_map_t>());
    index_map_t& remap_to_glob_row = *(remap_to_glob_row_vec.back().get());

    {
      std::vector<Str> row_names(0);
      CHECK(read_vector_file(row_file, row_names));

      std::vector<Index> local_index(row_names.size());  // original
      std::vector<Index> rel_local_index;                // relevant local indexes
      std::iota(local_index.begin(), local_index.end(), 0);
      std::copy_if(local_index.begin(), local_index.end(), std::back_inserter(rel_local_index),
                   [&](const Index i) { return is_glob_row(row_names.at(i)); });

      std::vector<index_pair_t> local_glob;  // local -> glob mapping

      std::transform(rel_local_index.begin(), rel_local_index.end(), std::back_inserter(local_glob),
                     [&](const Index _local) {
                       const Index _glob = _glob_row_pos(row_names.at(_local));
                       return std::make_pair(_local, _glob);
                     });

      remap_to_glob_row.insert(local_glob.begin(), local_glob.end());
    }

    ////////////////////////////////
    // What are relevant columns? //
    ////////////////////////////////

    remap_to_glob_col_vec.push_back(std::make_shared<index_map_t>());
    index_map_t& remap_to_glob_col  = *(remap_to_glob_col_vec.back().get());
    index_map_t& remap_to_local_col = *(remap_to_local_col_vec.back().get());

    {
      col_counter_on_valid_rows_t counter(remap_to_glob_row);
      visit_matrix_market_file(mtx_file, counter);
      const Vec& nnz_col = counter.Col_N;

      std::vector<Str> column_names(0);
      CHECK(read_vector_file(col_file, column_names));

      ASSERT(column_names.size() >= counter.max_col, "Insufficient number of columns");

      std::vector<Index> cols(counter.max_col);
      std::iota(std::begin(cols), std::end(cols), 0);
      std::vector<Index> valid_cols;
      std::copy_if(cols.begin(), cols.end(), std::back_inserter(valid_cols),
                   [&](const Index j) { return nnz_col(j) >= column_threshold; });

      TLOG("Found " << valid_cols.size() << " (with the sum >=" << column_threshold << ")");

      std::vector<Index> idx(valid_cols.size());
      std::vector<Index> glob_cols(valid_cols.size());
      std::iota(glob_cols.begin(), glob_cols.end(), glob_max_col);

      auto fun_local2glob = [&](const Index j) {
        return std::make_pair(valid_cols.at(j), glob_cols.at(j));
      };
      auto fun_glob2local = [&](const Index j) {
        return std::make_pair(glob_cols.at(j), valid_cols.at(j));
      };
      std::vector<index_pair_t> local2glob;
      std::vector<index_pair_t> glob2local;

      std::transform(idx.begin(), idx.end(), std::back_inserter(local2glob), fun_local2glob);
      std::transform(idx.begin(), idx.end(), std::back_inserter(local2glob), fun_glob2local);

      remap_to_glob_col.insert(local2glob.begin(), local2glob.end());
      remap_to_local_col.insert(glob2local.begin(), glob2local.end());

      glob_max_col += glob_cols.size();  // cumulative
    }

    TLOG("Created valid column names");
  }

  //////////////////////////////
  // create merged data files //
  //////////////////////////////

  TLOG("Start writing the merged data set");

  for (int b = 4; b < argc; b += 3) {
    const Str mtx_file(argv[b]);
    const Str row_file(argv[b + 1]);
    const Str col_file(argv[b + 2]);

    TLOG("MTX : " << mtx_file);
    TLOG("ROW : " << row_file);
    TLOG("COL : " << col_file);

    const Index batch_index              = (b - 4);
    const index_map_t& remap_to_glob_row = *(remap_to_glob_row_vec.at(batch_index).get());
    const index_map_t& remap_to_glob_col = *(remap_to_glob_col_vec.at(batch_index).get());
  }

  //   TLOG("Imported all the data files");

  //   Str output_mtx    = output + ".mtx.gz";
  //   Str output_column = output + ".columns.gz";

  //   TLOG("Output the columns --> " << output_column);
  //   write_pair_file(output_column, univ_column_names);

  //   TLOG("Number of non-zero elements = " << univ_Tvec.size());
  //   {
  //     ogzstream ofs(output_mtx.c_str(), std::ios::out);

  //     ofs.precision(4);

  //     ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
  //     ofs << univ_max_row << " ";
  //     ofs << univ_max_col << " ";
  //     ofs << univ_Tvec.size() << std::endl;

  //     ///////////////////////////////////////////////
  //     // note: matrix market uses 1-based indexing //
  //     ///////////////////////////////////////////////

  //     Index i, j;
  //     Scalar v;

  //     const Index INTERVAL    = 1e6;
  //     const Index max_triples = univ_Tvec.size();
  //     const Index MAX_PRINT   = (max_triples / INTERVAL);
  //     Index _num_triples      = 0;

  //     for (auto tt : univ_Tvec) {
  //       std::tie(i, j, v) = tt;
  //       ofs << (i + 1) << " " << (j + 1) << " " << v << std::endl;

  //       if (++_num_triples % INTERVAL == 0) {
  //         const Index _wrote = (_num_triples / INTERVAL);
  //         std::cerr << "\r";
  //         std::cerr << std::left << std::setfill('.') << std::setw(30) << "Writing ";
  //         std::cerr << std::setfill(' ') << std::setw(10) << _wrote;
  //         std::cerr << " x 1M (total " << std::setw(10) << MAX_PRINT << ")\r";
  //         std::cerr << std::flush;
  //       }
  //     }

  //     ofs.close();
  //   }
  //   TLOG("Wrote " << output_mtx << " file");

  //   Str output_row = output + ".rows.gz";
  //   write_vector_file(output_row, univ_rows);

  TLOG("Successfully finished");
  return EXIT_SUCCESS;
}
