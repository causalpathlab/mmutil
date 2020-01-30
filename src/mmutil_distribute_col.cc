#include "mmutil_distribute_col.hh"

void
print_help(const char* fname) {

  const char* _desc =
      "[Arguments]\n"
      "MTX:        Input matrix market file\n"
      "MEMBERSHIP: Input membership file for the columns\n"
      "OUTPUT:     Output file header\n"
      "\n";

  std::cerr << "Distribute columns into multiple filesets" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " MTX MEMBERSHIP OUTPUT" << std::endl;
  std::cerr << std::endl;
  std::cerr << _desc << std::endl;
}

int
main(const int argc, const char* argv[]) {

  if (argc < 4) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Str mtx_file(argv[1]);
  const Str membership_file(argv[2]);
  const Str output(argv[3]);

  ERR_RET(!file_exists(mtx_file), "missing the mtx file");
  ERR_RET(!file_exists(membership_file), "missing the mtx file");

  distribute_col(mtx_file, membership_file, output);

  TLOG("Successfully distributed to all the data files.");
  return EXIT_SUCCESS;
}
