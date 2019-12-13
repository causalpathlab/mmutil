#include "mmutil.hh"
#include "mmutil_stat.hh"

void print_help(const char* fname) {
  std::cerr << "Filter in informative features to reduce computational cost" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " top_features mtx_file feature_file output" << std::endl;
  std::cerr << std::endl;
  std::cerr << "We prioritize features by their prevalence across samples." << std::endl;
  std::cerr << std::endl;
}

////////////////////////
// compute statistics //
////////////////////////

inline auto compute_mtx_stat_file(const std::string filename) {
  row_stat_collector_t collector;
  visit_matrix_market_file(filename, collector);

  /////////////////////////////////
  // Unbiased standard deviation //
  /////////////////////////////////

  const Vec& s1  = collector.Row_S1;
  const Vec& s2  = collector.Row_S2;
  const Scalar n = static_cast<Scalar>(collector.max_col);

  Vec ret = s2 - s1.cwiseProduct(s1 / n);
  ret     = ret / std::max(n - 1.0, 1.0);
  ret     = ret.cwiseSqrt();

  return std::make_tuple(ret, collector.max_row, collector.max_col);
}

//////////
// main //
//////////

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str         = std::string;
  using copier_t    = triplet_copier_remapped_rows_t<Index, Scalar>;
  using index_map_t = copier_t::index_map_t;

  const Index Ntop = boost::lexical_cast<Index>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str feature_file(argv[3]);
  const Str output(argv[4]);

  ////////////////////////////////
  // First calculate row scores //
  ////////////////////////////////

  Vec row_scores;
  Index max_row, max_col;
  std::tie(row_scores, max_row, max_col) = compute_mtx_stat_file(mtx_file);

  /////////////////////
  // Prioritize rows //
  /////////////////////

  auto order = eigen_argsort_descending(row_scores);

  TLOG("row scores: " << row_scores(order.at(0)) << " ~ "
                      << row_scores(order.at(order.size() - 1)));

  //////////////////////////////
  // output selected features //
  //////////////////////////////

  std::vector<Str> features(0);
  CHECK(read_vector_file(feature_file, features));

  const Index Nout = std::min(Ntop, max_row);
  std::vector<Str> out_features(Nout);
  std::vector<Scalar> out_scores(Nout);
  std::vector<Scalar> out_full_scores(max_row);
  index_map_t remap;

  std::vector<Index> index_top(Nout);
  std::iota(std::begin(index_top), std::end(index_top), 0);
  std::for_each(index_top.begin(), index_top.end(),  //
                [&](const Index i) {
                  const Index j   = order.at(i);
                  out_features[i] = features.at(j);
                  out_scores[i]   = row_scores(j);
                  remap[j]        = i;
                });

  std::vector<Index> index_all(max_row);
  std::iota(std::begin(index_all), std::end(index_all), 0);

  std::for_each(index_all.begin(), index_all.end(),  //
                [&](const Index i) {
                  const Index j      = order.at(i);
                  out_full_scores[i] = row_scores(j);
                });

  Str output_mtx_file = output + ".mtx.gz";
  copier_t copier(output_mtx_file, remap);
  visit_matrix_market_file(mtx_file, copier);

  Str output_feature_file    = output + ".rows.gz";
  Str output_score_file      = output + ".scores.gz";
  Str output_full_score_file = output + ".full_scores.gz";

  write_vector_file(output_feature_file, out_features);
  write_vector_file(output_score_file, out_scores);
  write_vector_file(output_full_score_file, out_full_scores);

  return EXIT_SUCCESS;
}
