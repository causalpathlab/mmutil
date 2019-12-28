#include "mmutil_rand_conv_col.hh"

void print_help(const char* fname) {
  const char* _desc =
      "[Arguments]\n"
      "N      : Sample size of generated data\n"
      "D      : Number of reference columns per sample\n"
      "MTX    : Reference sparse matrix file (MTX)\n"
      "OUTPUT : Output file header\n"
      "\n";

  std::cerr << "Generate \"bulk\" data aggregating columns" << std::endl;
  std::cerr << fname << " N D MTX OUTPUT\n" << std::endl;
  std::cerr << _desc << std::endl;
}

int main(const int argc, const char* argv[]) {

  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str     = std::string;
  const Index N = std::stoi(argv[1]);
  const Index D = std::stoi(argv[2]);
  const Str mtx_file(argv[3]);
  const Str output(argv[4]);

  ERR_RET(!file_exists(mtx_file), "missing the mtx file");
  const SpMat RefMat = build_eigen_sparse(mtx_file);

  // sample triplets
  const Index Nref = RefMat.cols();

  const SpMat C = sample_conv_index(NrefT(Nref),     //
                                    ConvSampleT(N),  //
                                    RefPerSampleT(D));

  Mat Y(RefMat.rows(), N);
  Y = Mat(RefMat * Y);

  Str out_data_file = output + ".conv.gz";
  write_data_file(out_data_file, Y);

  Str out_mix_file = output + ".mix.gz";
  write_data_file(out_mix_file, C);

  return EXIT_SUCCESS;
}
