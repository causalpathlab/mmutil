#include <iostream>
#include "utils/util.hh"
#include "utils/io.hh"

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Sparse"

bool exists(const std::string fname) {
  using namespace boost::filesystem;



}



// Merge two large matrix market files

// Uniform 

void print_help(const char* fname) {
  std::cerr << fname << " {merge,filter}" << std::endl;
}

int main(const int argc, const char *argv[]) {

  if(argc < 3) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  // std::string filename(argv[1]);

  std::string mtx_file(argv[1]);
  std::string row_name_file(argv[2]);

  std::vector<std::string> row_names(0);

  auto _rows = read_vector_file(row_name_file, row_names);

  using Scalar = float;

  Eigen::SparseMatrix<Scalar> A;

  auto _data = read_matrix_market_file<Scalar>(mtx_file, A);


  // if(ret == EXIT_SUCCESS) {
  //   for(auto s : row_names){
  //     std::cout << s << std::endl;
  //   }
  // }




  // TLOG("Reading : " << filename);

  // 1. Read two sparse matrices

  // 2. Identify common rows or union rows

  // 3. Merge two 

  // 4. (Optional) Identify connected components based on shared-neighborhood graphs

  // Maybe some columns are isolated

  return EXIT_SUCCESS;
}


