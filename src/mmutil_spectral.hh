#include "mmutil.hh"
#include "svd.hh"

#ifndef MMUTIL_SPECTRAL_HH_
#define MMUTIL_SPECTRAL_HH_

template <typename Derived>
SpMat normalize_to_median(const Eigen::SparseMatrixBase<Derived>& xx) {

  const Derived& X                           = xx.derived();
  const Vec deg                              = X.transpose() * Mat::Ones(X.cols(), 1);
  std::vector<typename Derived::Scalar> _deg = std_vector(deg);
  TLOG("search the median degree [0, " << _deg.size() << ")");
  std::nth_element(_deg.begin(), _deg.begin() + _deg.size() / 2, _deg.end());
  const Scalar median = std::max(_deg[_deg.size() / 2], static_cast<Scalar>(1.0));

  TLOG("Targeting the median degree " << median);

  const Vec degInverse = deg.unaryExpr([&median](const Scalar x) {
    const Scalar _one = 1.0;
    return median / std::max(x, _one);
  });

  SpMat ret(X.rows(), X.cols());
  ret = X * degInverse.asDiagonal();

  return ret;
}

template <typename Derived>
auto take_spectrum_laplacian(                     //
    const Eigen::SparseMatrixBase<Derived>& _X0,  // sparse data
    const float tau_scale,                        // regularization
    const int rank,                               // desired rank
    const int iter = 5                            // should be enough
) {
  /////////////////////////////////////////////
  // 1. construct normalized scaled data     //
  // 2. identify eigen spectrum by using SVD //
  /////////////////////////////////////////////

  const Derived& X0 = _X0.derived();

  using Scalar = typename Derived::Scalar;
  using Index  = typename Derived::Index;
  using Mat    = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  {
    Mat xx = Mat(X0);
    std::cout << xx.cwiseProduct(xx).sum() << std::endl;
  }

  const Index max_col = X0.cols();
  const Index max_row = X0.rows();

  TLOG("Constructing a reguarlized graph Laplacian ...");

  const SpMat X    = normalize_to_median(X0);
  const Mat Deg    = (X.transpose().cwiseProduct(X.transpose()) * Mat::Ones(max_row, 1));
  const Scalar tau = Deg.mean() * tau_scale;
  const Mat degree_tau_sqrt_inverse = Deg.unaryExpr([&tau](const Scalar x) {
    const Scalar _one = 1.0;
    return _one / std::max(_one, std::sqrt(x + tau));
  });

  const Mat XtTau = degree_tau_sqrt_inverse.asDiagonal() * (X.transpose());

  TLOG("Running SVD on X [" << XtTau.rows() << " x " << XtTau.cols() << "]");

  RandomizedSVD<Mat> svd(rank, iter);
  svd.compute(XtTau);

  const Mat U = svd.matrixU();
  const Mat V = svd.matrixV();
  const Vec D = svd.singularValues();

  return std::make_tuple(U, V, D);
}

#endif
