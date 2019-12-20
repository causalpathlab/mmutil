#include "mmutil_rand_col.hh"

void print_help(const char* fname) {
  std::cerr << "Randomly select columns (samples) to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " num_col mtx_file sample_file output" << std::endl;
  std::cerr << std::endl;
}

////////////////////////
// compute statistics //
////////////////////////

inline auto compute_mtx_stat_file(const std::string filename) {
  col_stat_collector_t collector;
  visit_matrix_market_file(filename, collector);

  const Vec& s1  = collector.Col_S1;
  const Vec& s2  = collector.Col_S2;
  const Scalar n = static_cast<Scalar>(collector.max_row);
  Vec _ret       = s2 - s1.cwiseProduct(s1 / n);
  _ret           = _ret / std::max(n - 1.0, 1.0);
  _ret           = _ret.cwiseSqrt();

  std::vector<Scalar> ret(_ret.size());
  std_vector(_ret, ret);
  return std::make_tuple(ret, collector.max_row, collector.max_col);
}

int main(const int argc, const char* argv[]) {

  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str      = std::string;
  using copier_t = triplet_copier_remapped_cols_t<Index, Scalar>;

  const Index Nsample = boost::lexical_cast<Index>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str column_file(argv[3]);
  const Str output(argv[4]);

  ERR_RET(!file_exists(mtx_file), "missing the mtx file");
  ERR_RET(!file_exists(column_file), "missing the column file");
  std::vector<Str> column_names(0);
  CHECK(read_vector_file(column_file, column_names));

  std::vector<Scalar> col_var;
  Index max_row, max_col;

  std::tie(col_var, max_row, max_col) = compute_mtx_stat_file(mtx_file);




  return EXIT_SUCCESS;
}
