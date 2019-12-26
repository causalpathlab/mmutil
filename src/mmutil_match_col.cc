#include "mmutil_match.hh"

void print_help(const char* fname) {

  const char* _desc =
      "[Arguments]\n"
      "SRC_MTX	:    Source MTX file\n"
      "SRC_COL	:    Source MTX file\n"
      "TGT_MTX	:    Target MTX file\n"
      "TGT_COL	:    Target MTX file\n"
      "K	:    K nearest neighbors\n"
      "M	:    # of bidirectional links\n"
      "\n"
      "The number of bi-directional links created for every new element during construction.\n"
      "Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic\n"
      "dimensionality and/or high recall, while low M work better for datasets with low intrinsic\n"
      "dimensionality and/or low recalls.\n"
      "\n"
      "N	:    # nearest neighbor lists\n"
      "\n"
      "The size of the dynamic list for the nearest neighbors (used during the search). A higher \n"
      "value leads to more accurate but slower search. This cannot be set lower than the number \n"
      "of queried nearest neighbors k. The value ef of can be anything between k and the size of \n"
      "the dataset.\n"
      "\n"
      "OUTPUT	:    Output file header\n"
      "\n"
      "[Reference]\n"
      "Malkov, Yu, and Yashunin. `Efficient and robust approximate nearest neighbor search using\n"
      "Hierarchical Navigable Small World graphs.` preprint: https://arxiv.org/abs/1603.09320\n"
      "\n"
      "See also:\n"
      "https://github.com/nmslib/hnswlib\n"
      "\n";

  std::cerr << "Find k-nearest neighbors of the source columns among the target data." << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " SRC_MTX SRC_COL TGT_MTX TGT_COL K M N OUTPUT" << std::endl;
  std::cerr << std::endl;
  std::cerr << _desc << std::endl;
}

int main(const int argc, const char* argv[]) {

  if (argc < 9) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string mtx_src_file(argv[1]);
  const std::string col_src_file(argv[2]);

  const std::string mtx_tgt_file(argv[3]);
  const std::string col_rgt_file(argv[4]);

  std::vector<std::tuple<Index, Index, Scalar> > out_index;

  const int knn = search_knn(SrcRowsT(build_eigen_sparse(mtx_src_file).transpose().eval()),  //
                             TgtRowsT(build_eigen_sparse(mtx_tgt_file).transpose().eval()),  //
                             KNN(argv[5]),                                                   //
                             BILINKS(argv[6]),                                               //
                             NNLIST(argv[7]),                                                //
                             out_index);

  CHK_ERR_RET(knn, "Failed to search kNN");

  const std::string out_file(argv[8]);

  write_tuple_file(out_file, out_index);

  return EXIT_SUCCESS;
}
