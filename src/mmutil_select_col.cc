#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

#include "mmutil_select.hh"

void
print_help(const char* fname) {

  const char* _desc =
      "[Arguments]\n"
      "MTX:       [Input] Matrix market file\n"
      "COLUMN:    [Input] Column name file\n"
      "SELECTED:  [Input] Column names to be selected\n"
      "OUTPUT:    [Output] File header\n"
      "\n";

  std::cerr << "Select a subset of columns" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " MTX COLUMN SELECTED OUTPUT" << std::endl;
  std::cerr << std::endl;
  std::cerr << _desc << std::endl;
}

int
main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;
  const Str mtx_file(argv[1]);
  const Str column_file(argv[2]);
  const Str select_file(argv[3]);
  const Str output(argv[4]);

  ERR_RET(!file_exists(mtx_file), "missing the MTX file");
  ERR_RET(!file_exists(column_file), "missing the COLUMN file");
  ERR_RET(!file_exists(select_file), "missing the SELECTED file");

  copy_selected_columns(mtx_file, column_file, select_file, output);

  return EXIT_SUCCESS;
}
