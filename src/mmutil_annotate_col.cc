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

  std::vector<std::string> columns;
  CHECK(read_vector_file(options.col, columns));

  ///////////////////////////////////////////////
  // step 1 : Initial sampling for pretraining //
  ///////////////////////////////////////////////

  row_col_stat_collector_t stat;
  visit_matrix_market_file(options.mtx, stat);

  ASSERT(stat.max_col <= columns.size(), "Needs column names");

  std::vector<Index> valid_rows;
  std::vector<Index> subcols;

  std::tie(valid_rows, subcols) = select_rows_columns(Ltot, stat, options);

  SpMat X0 = read_eigen_sparse_subset_col(options.mtx, subcols);
  Mat L = Mat(row_sub(Ltot, valid_rows));
  Mat X = Mat(row_sub(X0, valid_rows));

  TLOG("Preprocessing X [" << X.rows() << " x " << X.cols() << "]");

  auto log2_op = [](const Scalar& x) -> Scalar { return std::log2(1.0 + x); };

  if (options.log_scale) {
    X = X.unaryExpr(log2_op);
  }

  normalize_columns(X);

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
  Vec sj(mu.cols());

  const Scalar eps = 1e-8;

  for (Index lb = 0; lb < N; lb += batch_size) {

    const Index ub = std::min(N, batch_size + lb);
    std::vector<Index> subcols_b(ub - lb);

    std::iota(subcols_b.begin(), subcols_b.end(), lb);

    if (options.verbose) TLOG("On the batch [" << lb << ", " << ub << ")");

    SpMat x0_b = read_eigen_sparse_subset_col(options.mtx, subcols_b);
    Mat xx_b = Mat(row_sub(x0_b, valid_rows));

    TLOG("Preprocessing xx [" << xx_b.rows() << " x " << xx_b.cols() << "]");

    Mat _col_norm = Mat::Ones(1, xx_b.rows()) * xx_b.cwiseProduct(xx_b);
    _col_norm.transposeInPlace();

    if (options.log_scale) {
      xx_b = xx_b.unaryExpr(log2_op);
    }

    normalize_columns(xx_b);

    if (options.verbose) TLOG("Prediction by argmax assignment");

    for (Index j = 0; j < xx_b.cols(); ++j) {

      sj = mu.transpose() * xx_b.col(j);

      if (_col_norm(j) < eps) continue;

      Index argmax;
      const Scalar score = sj.maxCoeff(&argmax);
      const Index k = subcols_b.at(j);

      output.emplace_back(columns.at(k), labels.at(argmax), score);
    }
  }

  std::vector<std::string> markers;
  markers.reserve(valid_rows.size());
  std::for_each(valid_rows.begin(), valid_rows.end(),
                [&](const auto r) { markers.emplace_back(rows.at(r)); });

  write_tuple_file(options.out + ".annot.gz", output);
  write_vector_file(options.out + ".marker_names.gz", markers);
  write_vector_file(options.out + ".label_names.gz", labels);

  return EXIT_SUCCESS;
}
