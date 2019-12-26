#include "mmutil.hh"
#include "mmutil_spectral.hh"

void print_help(const char* fname) {
  std::cerr << "Find an eigen spectrum of regularized graph Laplacian" << std::endl;
  std::cerr << "Output [U,D,V] of the corresponding SVD" << std::endl;
  std::cerr << "Ref. Qin and Rohe (2013)" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " tau_scale rank mtx_file output" << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Scalar tau_scale = std::stof(argv[1]);
  const Index rank       = std::stoi(argv[2]);
  const Str mtx_file(argv[3]);
  const Str output(argv[4]);

  const Index iter = 5;  // should be enough

  ///////////////////
  // Read the data //
  ///////////////////

  SpMat X0 = build_eigen_sparse(mtx_file);

  Mat U, V, D;
  std::tie(U, V, D) = take_spectrum_laplacian(X0, tau_scale, rank, iter);

  const Str output_U_file = output + ".u.gz";
  const Str output_V_file = output + ".v.gz";
  const Str output_D_file = output + ".d.gz";

  TLOG("Output results");

  write_data_file(output_U_file, U);
  write_data_file(output_V_file, V);
  write_data_file(output_D_file, D);

  TLOG("Done");

  return EXIT_SUCCESS;
}
