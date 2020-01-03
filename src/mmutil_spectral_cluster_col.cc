#include "mmutil_spectral_cluster_col.hh"

int
main(const int argc, const char* argv[]) {

  using F0 = trunc_dpm_t<Mat>;
  using F  = multi_gaussian_component_t<Mat>;
  using std::tie;
  using std::tuple;
  using std::vector;

  cluster_options_t options;

  CHECK(parse_cluster_options(argc, argv, options));

  const std::string mtx_file(options.mtx);
  const std::string col_file(options.col);

  ERR_RET(!file_exists(mtx_file), "No MTX data file");

  const std::string output(options.out);

  const SpMat X = build_eigen_sparse(mtx_file);
  const Index N = X.cols();
  Mat Data      = create_clustering_data(X, options);

  Mat Z, C;
  vector<Scalar> elbo;
  tie(Z, C, elbo) = estimate_mixture_of_columns<F0, F>(Data, options);

  write_data_file(output + ".centroid.gz", C);

  vector<Scalar> count = std_vector(Z * Mat::Ones(N, 1));
  print_histogram(count);

  if (file_exists(col_file)) {
    std::vector<std::string> samples;
    CHECK(read_vector_file(col_file, samples));
    auto argmax = create_argmax_vector(Z, samples);
    write_tuple_file(output + ".argmax.gz", argmax);
  } else {
    std::vector<Index> samples(N);
    std::iota(samples.begin(), samples.end(), 0);
    auto argmax = create_argmax_vector(Z, samples);
    write_tuple_file(output + ".argmax.gz", argmax);
  }

  if(options.out_data) {
    write_data_file(output + ".data.gz", Data);
  }

  return EXIT_SUCCESS;
}
