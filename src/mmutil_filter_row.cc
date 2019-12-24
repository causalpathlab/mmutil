#include "mmutil_filter_row.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative features to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " top_features mtx_file feature_file output" << std::endl;
  std::cerr << std::endl;
  std::cerr << "We prioritize features by their prevalence across samples." << std::endl;
  std::cerr << std::endl;
}

//////////
// main //
//////////

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str         = std::string;
  using copier_t    = triplet_copier_remapped_rows_t<Index, Scalar>;
  using index_map_t = copier_t::index_map_t;

  const Index Ntop = boost::lexical_cast<Index>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str feature_file(argv[3]);
  const Str output(argv[4]);

  ERR_RET(!file_exists(mtx_file), "missing the mtx file");
  ERR_RET(!file_exists(feature_file), "missing the feature file");

  filter_row_by_sd(Ntop, mtx_file, feature_file, output);

  return EXIT_SUCCESS;
}
