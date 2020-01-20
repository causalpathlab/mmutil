#include "mmutil.hh"
#include "mmutil_normalize.hh"
#include "svd.hh"

#ifndef MMUTIL_SPECTRAL_HH_
#define MMUTIL_SPECTRAL_HH_

////////////////////////////////////////////////////////////////////////////////
// Why is this graph Laplacian?
//
// (1) We let adjacency matrix A = X'X assuming elements in X are non-negative
//
//
// (2) Let the Laplacian L = I - D^{-1/2} A D^{-1/2}
//                         = I - D^{-1/2} (X'X) D^{-1/2}
///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
inline Mat
make_scaled_regularized(
    const Eigen::SparseMatrixBase<Derived>& _X0,  // sparse data
    const float tau_scale,                        // regularization
    const bool log_trans = true                   // log-transformation
) {

  const Derived& X0 = _X0.derived();

  using Scalar = typename Derived::Scalar;
  using Index  = typename Derived::Index;
  using Mat    = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  const Index max_row = X0.rows();

  auto trans_fun = [&log_trans](const Scalar& x) -> Scalar {
    if (x < 0.0) return 0.;
    return log_trans ? fasterlog(x + 1.0) : x;
  };

  const SpMat X = normalize_to_median(X0).unaryExpr(trans_fun);

  TLOG("Constructing a reguarlized graph Laplacian ...");

  const Mat Deg =
      (X.transpose().cwiseProduct(X.transpose()) * Mat::Ones(max_row, 1));
  const Scalar tau = Deg.mean() * tau_scale;

  const Mat degree_tau_sqrt_inverse = Deg.unaryExpr([&tau](const Scalar& x) {
    const Scalar _one = 1.0;
    return _one / std::max(_one, std::sqrt(x + tau));
  });

  Mat ret = degree_tau_sqrt_inverse.asDiagonal() * (X.transpose());
  return ret;  // RVO
}

// Batch-normalized graph Laplacian
// * Apply weights on features (rows; genes) to X matrix.

template <typename Derived, typename Derived2>
inline Mat
make_feature_normalized_laplacian(
    const Eigen::SparseMatrixBase<Derived>& _X0,  // sparse data
    const Eigen::MatrixBase<Derived2>& _weights,  // row weights
    const float tau_scale,                        // regularization
    const bool log_trans = true                   // log-transformation
) {

  const Derived& X0       = _X0.derived();
  const Derived2& weights = _weights.derived();
  const Index max_row     = X0.rows();

  ASSERT(weights.rows() == max_row, "We need weights on each row");
  ASSERT(weights.cols() == 1, "Provide summary vector");

  auto trans_fun = [&log_trans](const Scalar& x) -> Scalar {
    if (x < 0.0) return 0.;
    return log_trans ? fasterlog(x + 1.0) : x;
  };

  const SpMat X = normalize_to_median(X0).unaryExpr(trans_fun);

  TLOG("Constructing a doubly-reguarlized graph Laplacian ...");

  ////////////////////////////////////////////////////////
  // make X(g,i) <- X(g,i) * min{1/sqrt(weight(g)),  1} //
  ////////////////////////////////////////////////////////

  auto _row_fun = [](const Scalar& x) -> Scalar {
    return x <= 0.0 ? 0.0 : std::sqrt(1.0 / x);
  };

  const Mat _rows_denom = weights.unaryExpr(_row_fun);

  //////////////////////////////////////////////
  // make X(g,i) <- X(g,i) / sqrt(D(i) + tau) //
  //////////////////////////////////////////////

  const Mat col_deg =
      (X.transpose().cwiseProduct(X.transpose()) * Mat::Ones(max_row, 1));

  const Scalar tau = col_deg.mean() * tau_scale;

  auto _col_fun = [&tau](const Scalar& x) -> Scalar {
    const Scalar _one = 1.0;
    return _one / std::max(_one, std::sqrt(x + tau));
  };

  const Mat _cols_denom = col_deg.unaryExpr(_col_fun);

  // normalize them

  Mat ret = _cols_denom * (X.transpose()) * _rows_denom;
  return ret;
}

/////////////////////////////////////////////
// 1. construct normalized scaled data     //
// 2. identify eigen spectrum by using SVD //
/////////////////////////////////////////////

template <typename Derived>
inline std::tuple<Mat, Mat, Mat>
take_spectrum_laplacian(                          //
    const Eigen::SparseMatrixBase<Derived>& _X0,  // sparse data
    const float tau_scale,                        // regularization
    const int rank,                               // desired rank
    const int iter = 5                            // should be enough
) {

  const Mat XtTau = make_scaled_regularized(_X0, tau_scale);

  TLOG("Running SVD on X [" << XtTau.rows() << " x " << XtTau.cols() << "]");

  RandomizedSVD<Mat> svd(rank, iter);
  svd.set_verbose();
  svd.compute(XtTau);

  TLOG("Done SVD");

  Mat U = svd.matrixU();
  Mat V = svd.matrixV();
  Vec D = svd.singularValues();

  return std::make_tuple(U, V, D);
}

#endif
