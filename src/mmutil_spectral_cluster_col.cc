#include "mmutil_spectral_cluster_col.hh"

int
main(const int argc, const char* argv[]) {

  cluster_options_t options;
  CHECK(parse_cluster_options(argc, argv, options));

  const std::string mtx_file(options.mtx);
  const std::string output(options.out);

  ERR_RET(!file_exists(options.mtx), "No MTX data file");

  Mat Data = create_clustering_data(options);

  if (options.method == cluster_options_t::DBSCAN) {
    run_dbscan(Data, options);
    return EXIT_SUCCESS;
  }

  run_mixture_model(Data, options);
  return EXIT_SUCCESS;
}
