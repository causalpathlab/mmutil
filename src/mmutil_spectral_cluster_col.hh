#include "mmutil.hh"
#include "mmutil_cluster.hh"
#include "mmutil_match.hh"
#include "mmutil_spectral.hh"
#include "svd.hh"

#ifndef MMUTIL_SPECTRAL_CLUSTER_COL_HH
#define MMUTIL_SPECTRAL_CLUSTER_COL_HH

template <typename TVEC>
inline SpMat
make_similarity_graph(const TVEC& knn_index, const Index N) {

  TLOG("Convert distance to similarity");
  std::vector<Eigen::Triplet<Scalar> > triplets;
  triplets.reserve(knn_index.size());

  auto _sim = [](const auto& tt) {
    const Index i  = std::get<0>(tt);
    const Index j  = std::get<1>(tt);
    const Scalar v = static_cast<Scalar>(1.0) - std::get<2>(tt);
    return Eigen::Triplet<Scalar>(i, j, v);
  };

  std::transform(knn_index.begin(), knn_index.end(), std::back_inserter(triplets), _sim);

  return build_eigen_sparse(triplets, N, N);
}

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

template <typename OFS>
void
print_histogram(const std::vector<Scalar>& nn, OFS& ofs, const Scalar height = 50.0,
                const Scalar cutoff = .01, const Index ntop = 20) {

  const Scalar ntot = std::accumulate(nn.begin(), nn.end(), 0.0);

  ofs << "<histogram>" << std::endl;

  auto _print = [&](const Index j) {
    const Scalar x = nn.at(j);
    ofs << std::setw(10) << (j) << " [" << std::setw(10) << std::floor(x) << "] ";
    for (int i = 0; i < std::floor(x / ntot * height); ++i) ofs << "*";
    ofs << std::endl;
  };

  auto _args = std_argsort(nn);

  if (_args.size() <= ntop) {
    std::for_each(_args.begin(), _args.end(), _print);
  } else {
    std::for_each(_args.begin(), _args.begin() + 20, _print);
  }
  ofs << "</histogram>" << std::endl;
}

#endif
