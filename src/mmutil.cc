#include <iostream>
#include "utils/util.hh"
#include "utils/io.hh"

bool exists(const std::stringn fname) {
  using namespace boost::filesystem;



}


// Merge two large matrix market files

// Uniform 

void print_help(const char* fname) {
  std::cerr << fname << " {merge,filter}" << std::endl;
}

int main(const int argc, const char *argv[]) {

  if(argc < 2) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  // std::string filename(argv[1]);



  // TLOG("Reading : " << filename);

  // 1. Read two sparse matrices

  // 2. Identify common rows or union rows

  // 3. Merge two 

  // 4. (Optional) Identify connected components based on shared-neighborhood graphs

  // Maybe some columns are isolated

  return EXIT_SUCCESS;
}


