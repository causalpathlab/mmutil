#include "mmutil.hh"
#include "mmutil_normalize.hh"
#include "svd.hh"

#ifndef MMUTIL_SPECTRAL_HH_
#define MMUTIL_SPECTRAL_HH_

/////////////////////////////////////////////
// 1. construct normalized scaled data     //
// 2. identify eigen spectrum by using SVD //
/////////////////////////////////////////////

template <typename Derived>
inline Mat
make_scaled_regularized(const Eigen::SparseMatrixBase<Derived>& _X0,  // sparse data
                        const float tau_scale){                        // regularization

  const Derived& X0 = _X0.derived();

  using Scalar = typename Derived::Scalar;
  using Index  = typename Derived::Index;
  using Mat    = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  const Index max_row = X0.rows();

  TLOG("Constructing a reguarlized graph Laplacian ...");

  const SpMat X= normalize_to_median(X0);

  const Mat Deg    = (X.transpose().cwiseProduct(X.transpose()) * Mat::Ones(max_row, 1));
  const Scalar tau = Deg.mean() * tau_scale;
  const Mat degree_tau_sqrt_inverse = Deg.unaryExpr([&tau](const Scalar x) {
    const Scalar _one = 1.0;
    return _one / std::max(_one, std::sqrt(x + tau));
  });

  Mat ret = degree_tau_sqrt_inverse.asDiagonal() * (X.transpose());
  return ret;  // RVO
}

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
