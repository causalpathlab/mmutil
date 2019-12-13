#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Distribute columns into multiple files to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " mtx_file membership_file output" << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 4) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Str mtx_file(argv[1]);
  const Str membership_file(argv[2]);
  const Str output(argv[3]);

  std::vector<Str> membership;
  std::vector<Str> columns;
  std::unordered_map<Str, Str> _column_membership;
  std::unordered_map<Str, Index> _batch_index;
  CHECK(read_pair_file(membership_file, _column_membership));

  Index k = 0;
  for (auto pp : _column_membership) {
    columns.push_back(pp.first);
    membership.push_back(pp.second);

    if (_batch_index.count(pp.second) < 1) {
      _batch_index[pp.second] = k++;
    }
  }

  std::vector<Str> _index_batch(_batch_index.size());

  for (auto pp : _batch_index) {
    _index_batch[pp.second] = pp.first;
  }

  for (auto pp : _batch_index) {
    TLOG("batches: " << pp.first << " " << pp.second)
  }

  const Index num_batches = _batch_index.size();

  using Triplet       = std::tuple<Index, Index, Scalar>;
  using TripletVec    = std::vector<Triplet>;
  using StrVec        = std::vector<Str>;
  using TripletVecVec = std::vector<TripletVec>;
  using StrVecVec     = std::vector<StrVec>;

  TripletVecVec data_batches(num_batches, TripletVec{});
  StrVecVec column_batches(num_batches, StrVec{});

  TLOG("Distribute the data into " << num_batches << " batches ");

  TripletVec Tvec;
  Index max_row, max_col;
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);

  const Index INTERVAL      = 1e6;
  const Index max_tvec_size = Tvec.size();
  Index _num_triples        = 0;

  for (auto tt : Tvec) {
    Index i, j;
    Scalar w;
    std::tie(i, j, w) = tt;

    if (j >= columns.size()) {
      WLOG("Found a column without a name: " << j);
      continue;
    }

    // const Str _batch = columns.at(j);
    const Index _index = _batch_index.at(membership.at(j));
    data_batches.at(_index).push_back(tt);
    column_batches.at(_index).push_back(columns.at(j));

    if (++_num_triples % INTERVAL == 0) {
      std::cerr << "\r" << std::left << std::setfill('.') << std::setw(30);
      std::cerr << "Partitioning " << std::right << std::setfill(' ');
      std::cerr << (_num_triples / INTERVAL) << " x 1M triplets (total ";
      std::cerr << std::setw(10) << (max_tvec_size / INTERVAL) << ")" << std::flush;
    }
  }

  std::cerr << std::endl;
  TLOG("Output data files");

  for (auto pp : _batch_index) {
    const Str batch_name(pp.first);
    const Index _index     = pp.second;
    const TripletVec& data = data_batches.at(_index);

    const SpMat X = build_eigen_sparse(data, max_row, max_col);

    TLOG("A sub-matrix for " << batch_name);

    std::vector<Index> valid_columns;
    SpMat subset_X;
    std::tie(subset_X, valid_columns) = filter_columns(X, 1);

    std::vector<Str> subset_columns;

    for (Index j : valid_columns) {
      subset_columns.push_back(columns.at(j));
    }

    Str output_mtx_file    = output + "_" + batch_name + ".mtx.gz";
    Str output_column_file = output + "_" + batch_name + ".columns.gz";

    write_vector_file(output_column_file, subset_columns);
    write_matrix_market_file(output_mtx_file, subset_X);
  }

  TLOG("Successfully finished distributing the data.");
  return EXIT_SUCCESS;
}
