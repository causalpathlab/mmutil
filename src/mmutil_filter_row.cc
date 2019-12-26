#include "mmutil_filter_row.hh"

void print_help(const char* fname) {

  const char* _desc =
      "[Arguments]\n"
      "TOP:       Number of top rows to select\n"
      "MTX:       Input matrix market file\n"
      "ROW:       Input row name file\n"
      "OUTPUT:    Output file header\n"
      "\n";

  std::cerr << "Filter in informative features to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " TOP MTX ROW OUTPUT" << std::endl;
  std::cerr << std::endl;
  std::cerr << _desc << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  const Index Ntop = std::stoi(argv[1]);
  const std::string mtx_file(argv[2]);
  const std::string feature_file(argv[3]);
  const std::string output(argv[4]);

  ERR_RET(!file_exists(mtx_file), "missing the mtx file");
  ERR_RET(!file_exists(feature_file), "missing the feature file");

  filter_row_by_sd(Ntop, mtx_file, feature_file, output);

  return EXIT_SUCCESS;
}
