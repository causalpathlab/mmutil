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
  using std::string;
  using std::tie;
  using std::tuple;
  using std::vector;

  const Scalar tau    = options.tau;
  const Index lu_iter = options.lu_iter;
  const Index rank    = options.rank;
  const Index N       = X.cols();
  const bool _log     = options.log_scale;

  RandomizedSVD<Mat> svd(rank, lu_iter);
  if (options.verbose) svd.set_verbose();

  if (file_exists(options.row_weight_file)) {
    TLOG("Apply inverse weights on the rows");
    const string weight_file = options.row_weight_file;
    vector<Scalar> _w;
    CHECK(read_vector_file(weight_file, _w));
    const Vec ww = eigen_vector(_w);

    ASSERT(ww.rows() == X.rows(), "Must have the same number of rows");
    const Mat xx = make_feature_normalized_laplacian(X, ww, tau, _log);
    svd.compute(xx);
    Mat Data = standardize(svd.matrixU()).transpose().eval();
    return Data;
  }

  const Mat xx = make_scaled_regularized(X, tau, _log);
  svd.compute(xx);
  Mat Data = standardize(svd.matrixU()).transpose().eval();
  return Data;
}

template <typename Derived, typename S>
inline std::vector<std::tuple<S, Index> >
create_argmax_pair(const Eigen::MatrixBase<Derived>& Z,
                   const std::vector<S>& samples) {

  ASSERT(Z.cols() == samples.size(),
         "#samples should correspond the columns of Z");

  auto _argmax = [&](const Index j) {
    Index ret;
    Z.col(j).maxCoeff(&ret);
    return std::make_tuple(samples.at(j), ret);
  };

  std::vector<Index> index(Z.cols());
  std::vector<std::tuple<S, Index> > membership;
  membership.reserve(Z.cols());
  std::iota(index.begin(), index.end(), 0);
  std::transform(index.begin(), index.end(), std::back_inserter(membership),
                 _argmax);

  return membership;
}

template <typename S>
inline std::vector<std::tuple<S, Index> >
create_argmax_pair(const std::vector<Index>& argmax,
                   const std::vector<S>& samples) {

  const Index N = argmax.size();
  ASSERT(N == samples.size(), "#samples should correspond the columns of Z");

  auto _argmax = [&](const Index j) {
    const Index k = argmax.at(j);
    return std::make_tuple(samples.at(j), k);
  };

  std::vector<Index> index(N);
  std::vector<std::tuple<S, Index> > ret;
  ret.reserve(N);
  std::iota(index.begin(), index.end(), 0);
  std::transform(index.begin(), index.end(), std::back_inserter(ret), _argmax);

  return ret;
}

template <typename Derived>
inline std::vector<Index>
create_argmax_vector(const Eigen::MatrixBase<Derived>& Z) {

  const Index N = Z.cols();
  std::vector<Index> ret;
  ret.reserve(N);

  auto _argmax = [&Z](const Index j) {
    Index _ret;
    Z.col(j).maxCoeff(&_ret);
    return _ret;
  };

  std::vector<Index> index(N);
  std::iota(index.begin(), index.end(), 0);
  std::transform(index.begin(), index.end(), std::back_inserter(ret), _argmax);

  return ret;
}

#endif
