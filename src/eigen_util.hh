#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <execution>
#include <functional>
#include <vector>
#include "utils/util.hh"

#ifndef EIGEN_UTIL_HH_
#define EIGEN_UTIL_HH_

template <typename EigenVec>
inline auto std_vector(const EigenVec& eigen_vec) {
  std::vector<typename EigenVec::Scalar> ret(eigen_vec.size());
  for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) {
    ret[j] = eigen_vec(j);
  }
  return ret;
}

template <typename EigenVec, typename StdVec>
inline void std_vector(const EigenVec& eigen_vec, StdVec& ret) {
  ret.resize(eigen_vec.size());
  using T = typename StdVec::value_type;
  for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) {
    ret[j] = static_cast<T>(eigen_vec(j));
  }
}

template <typename T>
inline auto eigen_triplets(const std::vector<T>& Tvec) {

  using Scalar      = float;
  using _Triplet    = Eigen::Triplet<Scalar>;
  using _TripletVec = std::vector<_Triplet>;

  _TripletVec ret;
  ret.reserve(Tvec.size());

  for (auto tt : Tvec) {
    ret.emplace_back(_Triplet(std::get<0>(tt), std::get<1>(tt), std::get<2>(tt)));
  }

  return ret;
}

template <typename Scalar>
inline auto eigen_triplets(const std::vector<Eigen::Triplet<Scalar> >& Tvec) {
  return Tvec;
}

template <typename TVEC, typename INDEX>
inline auto build_eigen_sparse(const TVEC& Tvec, const INDEX max_row, const INDEX max_col) {

  auto _tvec   = eigen_triplets(Tvec);
  using Scalar = float;
  using SpMat  = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;

  SpMat ret(max_row, max_col);
  ret.reserve(_tvec.size());
  ret.setFromTriplets(_tvec.begin(), _tvec.end());
  return ret;
}

template <typename Vec>
auto eigen_argsort_descending(const Vec& data) {
  using Index = typename Vec::Index;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::execution::seq, std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data(lhs) > data(rhs); });
  return index;
}

template <typename Vec>
auto eigen_argsort_descending_par(const Vec& data) {
  using Index = typename Vec::Index;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::execution::seq, std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data(lhs) > data(rhs); });
  return index;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> row_score_degree(
    const Eigen::SparseMatrixBase<Derived>& _xx) {

  const Derived& xx = _xx.derived();
  using Scalar      = typename Derived::Scalar;
  using Mat         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  return xx.unaryExpr([](const Scalar x) { return std::abs(x); }) * Mat::Ones(xx.cols(), 1);
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> row_score_sd(
    const Eigen::SparseMatrixBase<Derived>& _xx) {

  const Derived& xx = _xx.derived();
  using Scalar      = typename Derived::Scalar;
  using Vec         = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Mat         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  Vec s1         = xx * Mat::Ones(xx.cols(), 1);
  Vec s2         = xx.cwiseProduct(xx) * Mat::Ones(xx.cols(), 1);
  const Scalar n = xx.cols();
  Vec ret        = s2 - s1.cwiseProduct(s1 / n);
  ret            = ret / std::max(n - 1.0, 1.0);
  ret            = ret.cwiseSqrt();

  return ret;
}

template <typename Derived>
Eigen::SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t> vcat(
    const Eigen::SparseMatrixBase<Derived>& _upper,
    const Eigen::SparseMatrixBase<Derived>& _lower) {

  using Scalar         = typename Derived::Scalar;
  using Index          = typename Derived::Index;
  const Derived& upper = _upper.derived();
  const Derived& lower = _lower.derived();

  ASSERT(upper.cols() == lower.cols(), "mismatching columns in vcat");

  using _Triplet = Eigen::Triplet<Scalar>;

  std::vector<_Triplet> triplets;
  triplets.reserve(upper.nonZeros() + lower.nonZeros());

  using SpMat =
      typename Eigen::SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>;

  for (Index k = 0; k < upper.outerSize(); ++k) {
    for (typename SpMat::InnerIterator it(upper, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  for (Index k = 0; k < lower.outerSize(); ++k) {
    for (typename SpMat::InnerIterator it(lower, k); it; ++it) {
      triplets.emplace_back(upper.rows() + it.row(), it.col(), it.value());
    }
  }

  SpMat result(lower.rows() + upper.rows(), upper.cols());
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}

template <typename Derived>
Eigen::SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t> hcat(
    const Eigen::SparseMatrixBase<Derived>& _left, const Eigen::SparseMatrixBase<Derived>& _right) {

  using Scalar         = typename Derived::Scalar;
  using Index          = typename Derived::Index;
  const Derived& left  = _left.derived();
  const Derived& right = _right.derived();

  ASSERT(left.rows() == right.rows(), "mismatching rows in hcat");

  using _Triplet = Eigen::Triplet<Scalar>;

  std::vector<_Triplet> triplets;
  triplets.reserve(left.nonZeros() + right.nonZeros());

  using SpMat =
      typename Eigen::SparseMatrix<typename Derived::Scalar, Eigen::RowMajor, std::ptrdiff_t>;

  for (Index k = 0; k < left.outerSize(); ++k) {
    for (typename SpMat::InnerIterator it(left, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  for (Index k = 0; k < right.outerSize(); ++k) {
    for (typename SpMat::InnerIterator it(right, k); it; ++it) {
      triplets.emplace_back(it.row(), left.cols() + it.col(), it.value());
    }
  }

  SpMat result(left.rows(), left.cols() + right.cols());
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}

#endif
