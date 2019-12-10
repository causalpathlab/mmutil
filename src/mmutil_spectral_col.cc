#include "mmutil.hh"
#include "svd.hh"

void print_help(const char* fname) {
  std::cerr << "Find an eigen spectrum of regularized graph Laplacian" << std::endl;
  std::cerr << "Output [U,D,V] of the corresponding SVD" << std::endl;
  std::cerr << "Ref. Qin and Rohe (2013)" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " tau_scale rank mtx_file output" << std::endl;
  std::cerr << std::endl;
}

template <typename Derived>
SpMat normalize_to_median(const Eigen::SparseMatrixBase<Derived>& xx) {
  const Derived& X = xx.derived();
  const Vec deg = X.transpose() * Mat::Ones(X.cols(), 1);
  std::vector<typename Derived::Scalar> _deg = std_vector(deg);
  TLOG("search the median degree [0, " << _deg.size() << ")");
  std::nth_element(_deg.begin(), _deg.begin() + _deg.size() / 2, _deg.end());
  const Scalar median = _deg[_deg.size() / 2];

  TLOG("Targeting the median degree " << median);

  const Vec degInverse = deg.unaryExpr([&median](const Scalar x) {
    const Scalar _one = 1.0;
    return median / std::max(x, _one);
  });

  SpMat ret(X.rows(), X.cols());
  ret = degInverse.asDiagonal() * X;
  return ret;
}

template <typename Vec>
auto std_argsort(const Vec& data) {
  using Index = unsigned int;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data.at(lhs) > data.at(rhs); });
  return index;
}

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Scalar tau_scale = boost::lexical_cast<Scalar>(argv[1]);
  const Index rank = boost::lexical_cast<Index>(argv[2]);
  const Str mtx_file(argv[3]);
  const Str output(argv[4]);

  const Index iter = 5;  // should be enough

  ///////////////////
  // Read the data //
  ///////////////////

  using Triplet = std::tuple<Index, Index, Scalar>;
  using TripletVec = std::vector<Triplet>;
  TripletVec Tvec;
  Index max_row, max_col;
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);

  TLOG(max_row << " x " << max_col);

  SpMat X0 = build_eigen_sparse(Tvec, max_row, max_col);
  TLOG("Normalize columns ...");
  SpMat X = normalize_to_median(X0);

  TLOG("Constructing a reguarlized graph Laplacian ...");

  const SpMat Xt = X.transpose();  // sample x feature
  const Mat Deg = Xt.cwiseProduct(Xt) * Mat::Ones(max_row, 1);
  const Scalar tau = Deg.mean() * tau_scale;
  const Mat degree_tau_sqrt_inverse = Deg.unaryExpr([&tau](const Scalar x) {
    const Scalar _one = 1.0;
    return _one / std::max(_one, std::sqrt(x + tau));
  });

  const Mat XtTau = degree_tau_sqrt_inverse.asDiagonal() * Xt;

  TLOG("Running SVD on X [" << XtTau.rows() << " x " << XtTau.cols() << "]");

  RandomizedSVD<Mat> svd(rank, iter);
  svd.compute(XtTau);

  const Mat U = svd.matrixU();
  const Mat V = svd.matrixV();
  const Vec D = svd.singularValues();

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
