#include "mmutil.hh"
#include "mmutil_cluster.hh"
#include "mmutil_match.hh"
#include "mmutil_spectral.hh"
#include "svd.hh"

#ifndef MMUTIL_SPECTRAL_CLUSTER_COL_HH
#define MMUTIL_SPECTRAL_CLUSTER_COL_HH

inline Mat
create_clustering_data(const SpMat& X, const cluster_options_t& options) {

  using std::ignore;
  using std::tie;
  using std::tuple;
  using std::vector;

  const float tau     = options.tau;
  const Index lu_iter = options.lu_iter;
  const Index rank    = options.rank;
  const Index N       = X.cols();

  Mat U;
  tie(U, ignore, ignore) = take_spectrum_laplacian(X, tau, rank, lu_iter);

  Mat Data(U.cols(), U.rows());
  Data = standardize(U).transpose();
  return Data;
}

template <typename Derived, typename S>
std::vector<std::tuple<S, Index> >
create_argmax_vector(const Eigen::MatrixBase<Derived>& Z, const std::vector<S>& samples) {

  ASSERT(Z.cols() == samples.size(), "#samples should correspond the columns of Z");

  auto _argmax = [&](const Index j) {
    Index ret;
    Z.col(j).maxCoeff(&ret);
    return std::make_tuple(samples.at(j), ret);
  };

  std::vector<Index> index(Z.cols());
  std::vector<std::tuple<S, Index> > membership;
  membership.reserve(Z.cols());
  std::iota(index.begin(), index.end(), 0);
  std::transform(index.begin(), index.end(), std::back_inserter(membership), _argmax);

  return membership;
}


#endif
