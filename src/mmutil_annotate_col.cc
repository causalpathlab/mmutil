#include "mmutil_annotate_col.hh"

int
main(const int argc, const char* argv[]) {

  annotate_options_t options;

  CHECK(parse_annotate_options(argc, argv, options));

  //////////////////////////////////////////////////////////
  // Read the annotation information to construct initial //
  // type-specific marker gene profiles			  //
  //////////////////////////////////////////////////////////

  SpMat Ltot;  // gene x label
  std::vector<std::string> rows;
  std::vector<std::string> labels;

  std::tie(Ltot, rows, labels) = read_annotation_matched(options);

  if (options.verbose)
    for (auto l : labels) {
      TLOG("Annotation Labels: " << l);
    }

  std::vector<std::string> columns;
  CHECK(read_vector_file(options.col, columns));

  ///////////////////////////////////////////////
  // step 1 : Initial sampling for pretraining //
  ///////////////////////////////////////////////

  row_col_stat_collector_t stat;
  visit_matrix_market_file(options.mtx, stat);

  ASSERT(stat.max_col <= columns.size(), "Needs column names");

  auto cv_fun = [](const Scalar& v, const Scalar& m) -> Scalar {
    return std::sqrt(v) / (m + 1e-8);
  };

  std::vector<Index> valid_rows;
  Vec nn         = Ltot * Mat::Ones(Ltot.cols(), 1);
  Vec lab_size   = Ltot.transpose() * Mat::Ones(Ltot.rows(), 1);
  const Index sz = lab_size.minCoeff();

  TLOG("Selecting " << sz << " markers for each label");
  // Step 1a. select rows by standard deviation
  {
    const Scalar nn = static_cast<Scalar>(stat.max_col);
    const Scalar mm = std::max(nn - 1.0, 1.0);

    const Vec& s1 = stat.Row_S1;
    const Vec& s2 = stat.Row_S2;

    Vec mu     = s1 / nn;
    Vec row_sd = ((s2 - s1.cwiseProduct(mu)) / mm).cwiseSqrt();

    for (Index k = 0; k < Ltot.cols(); ++k) {  // each clust
      Vec l_k    = row_sd.cwiseProduct(Ltot.col(k));
      auto order = eigen_argsort_descending(l_k);
      std::copy(order.begin(), order.begin() + sz,
                std::back_inserter(valid_rows));
    }
  }

  std::vector<std::string> markers(valid_rows.size());
  std::for_each(valid_rows.begin(), valid_rows.end(),
                [&](const auto r) { markers.emplace_back(rows.at(r)); });

  if (options.verbose)
    for (auto r : valid_rows) {
      TLOG("Marker [" << r << "] " << rows.at(r));
    }

  Mat L = Mat(row_sub(Ltot, valid_rows));

  // Step 1b. Select columns by high coefficient of variance
  std::vector<Index> subcols;
  {
    const Scalar nn = static_cast<Scalar>(stat.max_row);
    const Scalar mm = std::max(nn - 1.0, 1.0);

    const Vec& s1 = stat.Col_S1;
    const Vec& s2 = stat.Col_S2;

    Vec mu     = s1 / nn;
    Vec col_cv = ((s2 - s1.cwiseProduct(mu)) / mm).binaryExpr(mu, cv_fun);

    std::vector<Index> index_r = eigen_argsort_descending(col_cv);
    const Index nsubsample     = std::min(stat.max_col, options.initial_sample);
    subcols.resize(nsubsample);
    std::copy(index_r.begin(), index_r.begin() + nsubsample, subcols.begin());
  }

  SpMat X0 = read_eigen_sparse_subset_col(options.mtx, subcols);
  Mat X    = Mat(row_sub(X0, valid_rows));

  TLOG("Preprocessing X [" << X.rows() << " x " << X.cols() << "]");

  auto log2_op = [](const Scalar& x) -> Scalar { return std::log2(0.5 + x); };

  if (options.log_scale) {
    X = X.unaryExpr(log2_op);
  }

  X.colwise().normalize();

  //////////////////////////////////////////
  // step2 : Train marker gene parameters //
  //////////////////////////////////////////

  TLOG("Fine-tuning marker gene parameters");

  Mat mu = train_marker_genes(L, X, options);

  /////////////////////////////////////////////////////
  // step3: Assign labels to all the cells (columns) //
  /////////////////////////////////////////////////////

  const Index batch_size = options.batch_size;

  using out_tup = std::tuple<std::string, std::string, Scalar>;
  std::vector<out_tup> output;
  const Index N = stat.max_col;

  output.reserve(N);

  for (Index lb = 0; lb < N; lb += batch_size) {

    const Index ub = std::min(N, batch_size + lb);
    std::vector<Index> subcols_b(ub - lb);

    std::iota(subcols_b.begin(), subcols_b.end(), lb);

    if (options.verbose) TLOG("On the batch [" << lb << ", " << ub << ")");

    SpMat x0_b = read_eigen_sparse_subset_col(options.mtx, subcols_b);
    Mat xx_b   = Mat(row_sub(x0_b, valid_rows));

    TLOG("Preprocessing xx [" << xx_b.rows() << " x " << xx_b.cols() << "]");

    if (options.log_scale) {
      xx_b = xx_b.unaryExpr(log2_op);
    }

    xx_b.colwise().normalize();

    if (options.verbose) TLOG("Prediction by argmax assignment");

    Mat scoreMat = mu.transpose() * xx_b;

    for (Index j = 0; j < scoreMat.cols(); ++j) {
      Index argmax;
      scoreMat.col(j).maxCoeff(&argmax);
      Scalar score  = scoreMat(argmax, j);
      const Index k = subcols_b.at(j);

      output.emplace_back(
          std::make_tuple(columns.at(k), labels.at(argmax), score));
    }
  }

  write_tuple_file(options.out + ".annot.gz", output);
  write_data_file(options.out + ".marker_data.gz", mu);
  write_vector_file(options.out + ".marker_names.gz", markers);
  write_vector_file(options.out + ".label_names.gz", labels);

  return EXIT_SUCCESS;
}
