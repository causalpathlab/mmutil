#include "mmutil.hh"
#include "mmutil_stat.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative samples to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " threshold mtx_file sample_file output" << std::endl;
  std::cerr << std::endl;
  std::cerr << "We prioritize samples by their prevalence across features." << std::endl;
  std::cerr << std::endl;
}

////////////////////////
// compute statistics //
////////////////////////

inline auto compute_mtx_stat_file(const std::string filename) {
  col_stat_collector_t collector;
  visit_matrix_market_file(filename, collector);
  Vec ret = collector.Col_N;
  return std::make_tuple(ret, collector.max_row, collector.max_col);
}

//////////
// main //
//////////

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str      = std::string;
  using copier_t = triplet_copier_remapped_cols_t<Index, Scalar>;

  const Scalar column_threshold = boost::lexical_cast<Scalar>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str column_file(argv[3]);
  const Str output(argv[4]);

  ERR_RET(!file_exists(mtx_file), "missing the mtx file");
  ERR_RET(!file_exists(column_file), "missing the column file");

  std::vector<Str> column_names(0);
  CHECK(read_vector_file(column_file, column_names));

  Vec nnz_col;
  Index max_row, max_col;
  std::tie(nnz_col, max_row, max_col) = compute_mtx_stat_file(mtx_file);

  ASSERT(column_names.size() >= max_col, "Insufficient number of columns");

  ///////////////////////////////////////////////////////
  // Filter out columns with too few non-zero elements //
  ///////////////////////////////////////////////////////

  std::vector<Index> cols(max_col);
  std::iota(std::begin(cols), std::end(cols), 0);
  std::vector<Index> valid_cols;
  std::copy_if(cols.begin(), cols.end(), std::back_inserter(valid_cols),
               [&](const Index j) { return nnz_col(j) >= column_threshold; });

  TLOG("Found " << valid_cols.size() << " (with the nnz >=" << column_threshold << ")");
  copier_t::index_map_t remap;

  std::vector<Str> out_column_names;
  std::vector<Index> index_out(valid_cols.size());
  std::vector<Scalar> out_scores;
  Index i   = 0;
  Index NNZ = 0;
  for (Index old_index : valid_cols) {
    remap[old_index] = i;
    out_column_names.push_back(column_names.at(old_index));
    out_scores.push_back(nnz_col(old_index));
    NNZ += nnz_col(old_index);
    ++i;
  }

  TLOG("Created valid column names");

  const Str output_column_file     = output + ".columns.gz";
  const Str output_full_score_file = output + ".full_scores.gz";
  const Str output_score_file      = output + ".scores.gz";
  const Str output_mtx_file        = output + ".mtx.gz";

  write_vector_file(output_column_file, out_column_names);

  auto out_full_scores = std_vector(nnz_col);
  write_vector_file(output_full_score_file, out_full_scores);

  write_vector_file(output_score_file, out_scores);

  copier_t copier(output_mtx_file, remap, NNZ);
  visit_matrix_market_file(mtx_file, copier);

  return EXIT_SUCCESS;
}
