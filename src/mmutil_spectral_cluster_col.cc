#include "mmutil_spectral_cluster_col.hh"

int
main(const int argc, const char* argv[]) {

  using std::string;
  using std::tie;
  using std::tuple;
  using std::vector;

  cluster_options_t options;

  CHECK(parse_cluster_options(argc, argv, options));

  const string mtx_file(options.mtx);
  const string col_file(options.col);

  ERR_RET(!file_exists(mtx_file), "No MTX data file");

  const string output(options.out);

  const SpMat X = build_eigen_sparse(mtx_file);
  const Index N = X.cols();
  Mat Data      = create_clustering_data(X, options);

  TLOG("Done with the initial spectral transformation");

  if (options.method != cluster_options_t::DBSCAN) {

    TLOG("Fitting a mixture model");

    Mat Z, C;
    using F0 = trunc_dpm_t<Mat>;
    using F  = multi_gaussian_component_t<Mat>;
    vector<Scalar> score;

    tie(Z, C, score) = estimate_mixture_of_columns<F0, F>(Data, options);

    ////////////////////////
    // output argmax file //
    ////////////////////////

    if (file_exists(col_file)) {
      vector<string> samples;
      CHECK(read_vector_file(col_file, samples));
      auto argmax = create_argmax_pair(Z, samples);
      write_tuple_file(output + ".argmax.gz", argmax);
    } else {
      vector<Index> samples(N);
      std::iota(samples.begin(), samples.end(), 0);
      auto argmax = create_argmax_pair(Z, samples);
      write_tuple_file(output + ".argmax.gz", argmax);
    }

    /////////////////////
    // show statistics //
    /////////////////////

    if (options.verbose) {
      Vec nn               = Z * Mat::Ones(N, 1);
      vector<Scalar> count = std_vector(nn);
      print_histogram(count, std::cout);
      std::cout << std::flush;
    }

    write_data_file(output + ".centroid.gz", C);

    if (options.out_data) {
      write_data_file(output + ".data.gz", Data);
    }

    //////////////////////////////////////
    // output low-dimensional embedding //
    //////////////////////////////////////

    TLOG("Embedding the clustering results");

    Mat xx, cc;
    vector<Index> argmax = create_argmax_vector(Z);
    tie(cc, xx)          = embed_by_centroid(Data, argmax, options);

    write_data_file(output + ".embedded.gz", xx);
    write_data_file(output + ".embedded_centroid.gz", cc);

    TLOG("Done fitting a mixture model");
    return EXIT_SUCCESS;
  }

  TLOG("Using Density-Based SCAN");

  vector<vector<Index> > membership;
  estimate_dbscan_of_columns(X, membership, options);

  // output cluster membership and embedding results
  Index l = 0;
  for (const vector<Index>& z : membership) {

    string _output = output + "_level_" + std::to_string(++l);

    if (file_exists(col_file)) {
      vector<string> samples;
      CHECK(read_vector_file(col_file, samples));
      auto argmax = create_argmax_pair(z, samples);
      write_tuple_file(_output + ".argmax.gz", argmax);
    } else {
      vector<Index> samples(N);
      std::iota(samples.begin(), samples.end(), 0);
      auto argmax = create_argmax_pair(z, samples);
      write_tuple_file(_output + ".argmax.gz", argmax);
    }

    Mat xx, cc;
    tie(cc, xx) = embed_by_centroid(Data, z, options);

    write_data_file(_output + ".embedded.gz", xx);
    write_data_file(_output + ".embedded_centroid.gz", cc);
  }

  return EXIT_SUCCESS;
}
