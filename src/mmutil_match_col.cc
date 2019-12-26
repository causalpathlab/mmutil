#include "mmutil_match.hh"

void print_help(const char* fname) {

  const char* _desc =
      "[Arguments]\n"
      "OUTPUT:    Output file header\n"
      "\n";

  std::cerr << "Filter in informative samples to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " SRC TGT OUTPUT" << std::endl;
  std::cerr << std::endl;
  std::cerr << _desc << std::endl;
}

// TODO:

// 1. reading through target and populate dictionary

// 2.

// 3. output

// src -> tgt with score

using Alg = hnswlib::HierarchicalNSW<Scalar>;

template <typename T>
void print_queue(T& q) {

  auto _print_pair = [](auto pp) {
    std::cout << "(" << std::get<0>(pp) << ", " << std::get<1>(pp) << ")";
  };

  while (!q.empty()) {
    _print_pair(q.top());
    std::cout << " ";
    q.pop();
  }
  std::cout << '\n';
}

int main(const int argc, const char* argv[]) {

  const std::string mtx_src_file(argv[1]);
  const std::string mtx_tgt_file(argv[2]);

  //   SpMat Tgt = build_eigen_matrix(mtx_tgt_file);

  //   // Each col = each data point and Eigen stores column-major (by default).
  //   // However, sparse matrix stores data in a row-major fashion.
  //   Mat TgtData(Tgt.rows(), Tgt.cols());
  //   TgtData = Mat(Tgt);

  //   const std::size_t vecdim  = TgtData.rows();
  //   const std::size_t vecsize = TgtData.cols();
  //   hnswlib::L2Space l2space(vecdim);

  //   const std::size_t param_bilinks = 2;
  //   const std::size_t param_nnlist = 10;

  //   Alg appr_alg(&l2space, vecsize, param_bilinks, param_nnlist);

  //   Scalar* mass = TgtData.data();

  //   for (Index i = 0; i < 1; ++i) {
  //     appr_alg.addPoint((void*)(mass + vecdim * i), static_cast<std::size_t>(i));
  //   }

  // #pragma omp parallel for
  //   for (Index i = 1; i < vecsize; ++i) {
  //     appr_alg.addPoint((void*)(mass + vecdim * i), static_cast<std::size_t>(i));
  //   }

  // TODO: Note data must be either stored or in the memory

  // appr_alg.ef_ = param_nnlist;

  ////////////
  // recall //
  ////////////

  // SpMat Src = build_eigen_matrix(mtx_src_file);

  // Mat SrcData(Src.rows(), Src.cols());
  // SrcData = Mat(Src);

  // const Scalar* src_mass = SrcData.data();

  // using labeltype = hnswlib::labeltype;

  // for (Index i = 0; i < SrcData.cols(); ++i) {

  //   std::priority_queue<std::pair<float, labeltype>> q =
  //       appr_alg.searchKnn((void*)(src_mass + vecdim * i), 10);

  //   print_queue(q);
  // }

  // .searchKnn(const void *query_data, size_t k)
}
