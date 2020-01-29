#include "mmutil.hh"
#include "mmutil_spectral.hh"

int
main(const int argc, const char* argv[]) {

  spectral_options_t options;
  CHECK(parse_spectral_options(argc, argv, options));
  ERR_RET(!file_exists(options.mtx), "No MTX data file");

  using Str = std::string;

  const Str mtx_file     = options.mtx;
  const Scalar tau_scale = options.tau;
  const Index rank       = options.rank;
  const Index iter       = options.iter;
  const Str output       = options.out;

  const Str wfile = options.row_weight_file;
  Vec weights;
  if (file_exists(wfile)) {
    std::vector<Scalar> ww;
    CHECK(read_vector_file(wfile, ww));
    weights = eigen_vector(ww);
  }

  ///////////////////
  // Read the data //
  ///////////////////

  Mat U, V, D;

  std::tie(U, V, D) = take_spectrum_nystrom(options.mtx,             //
                                            weights,                 //
                                            options.tau,             //
                                            options.col_norm,        //
                                            options.rank,            //
                                            options.iter,            //
                                            options.nystrom_sample,  //
                                            options.nystrom_batch,   //
                                            options.log_scale);

  // SpMat X0 = build_eigen_sparse(mtx_file);
  // std::tie(U, V, D) = take_spectrum_laplacian(X0, tau_scale, rank, iter);

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
