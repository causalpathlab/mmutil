#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative samples to reduce computational cost"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " threshold mtx_file sample_file output"
            << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "We prioritize samples by their prevalence across features."
      << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Scalar column_threshold = boost::lexical_cast<Scalar>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str column_file(argv[3]);
  const Str output(argv[4]);

  using Triplet = std::tuple<Index, Index, Scalar>;
  using TripletVec = std::vector<Triplet>;

  TripletVec Tvec;
  Index max_row, max_col;
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);

  std::vector<Str> columns(0);
  CHECK(read_vector_file(column_file, columns));

  ASSERT(columns.size() == max_col, "Data and Column names should match");

  // Calculate some scores on the sparse matrix
  const SpMat X0 = build_eigen_sparse(Tvec, max_row, max_col);
  SpMat out_X;
  std::vector<Index> valid_columns;
  std::tie(out_X, valid_columns) = filter_columns(X0, column_threshold);
  
  std::vector<Str> out_columns;
  for(Index k : valid_columns) {
    out_columns.push_back(columns.at(k));
  }

  Str output_mtx_file = output + ".mtx.gz";
  Str output_column_file = output + ".columns.gz";

  write_vector_file(output_column_file, out_columns);
  write_matrix_market_file(output_mtx_file, out_X);

  return EXIT_SUCCESS;
}
