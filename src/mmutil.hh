#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/lexical_cast.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "io.hh"
#include "io_visitor.hh"
#include "utils/util.hh"

#ifndef MMUTIL_HH_
#define MMUTIL_HH_

using Scalar = float;
using SpMat  = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;
using Index  = SpMat::Index;

using Mat = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename EigenVec>
inline auto std_vector(const EigenVec& eigen_vec) {
  std::vector<typename EigenVec::Scalar> ret(eigen_vec.size());
  for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) ret[j] = eigen_vec(j);
  return ret;
}

template <typename EigenVec, typename StdVec>
inline void std_vector(const EigenVec& eigen_vec, StdVec& ret) {
  ret.resize(eigen_vec.size());
  using T = typename StdVec::value_type;
  for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j)
    ret[j] = static_cast<T>(eigen_vec(j));
}

template <typename Vec>
auto std_argsort(const Vec& data) {
  using Index = std::ptrdiff_t;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::execution::par, std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data.at(lhs) > data.at(rhs); });
  // std::sort(std::begin(index), std::end(index),
  //           [&](Index lhs, Index rhs) { return data.at(lhs) > data.at(rhs); });
  return index;
}

template <typename TVEC>
inline auto build_eigen_triplets(const TVEC& Tvec) {
  using _Triplet    = Eigen::Triplet<Scalar>;
  using _TripletVec = std::vector<_Triplet>;

  _TripletVec ret(Tvec.size());

  const Index INTERVAL      = 1e6;
  const Index max_tvec_size = Tvec.size();

  Index j = 0;
  for (auto tt : Tvec) {
    Index r;
    Index c;
    Scalar w;
    std::tie(r, c, w) = tt;

    ret.push_back(_Triplet(r, c, w));

    if (++j % INTERVAL == 0) {
      std::cerr << "\r" << std::left << std::setfill('.');
      std::cerr << std::setw(30) << "Adding ";
      std::cerr << std::right << std::setfill(' ') << std::setw(10);
      std::cerr << (j / INTERVAL) << " x 1M triplets";
      std::cerr << " (total " << std::setw(10) << (max_tvec_size / INTERVAL) << ")";
      std::cerr << "\r" << std::flush;
    }
  }

  std::cerr << std::endl;
  return ret;
}

template <typename TVEC, typename INDEX>
inline auto build_eigen_sparse(const TVEC& Tvec, const INDEX max_row, const INDEX max_col) {
  auto _tvec = build_eigen_triplets(Tvec);

  SpMat ret(max_row, max_col);
  ret.reserve(_tvec.size());
  ret.setFromTriplets(_tvec.begin(), _tvec.end());
  return ret;
}

template <typename Derived>
auto filter_columns(const Eigen::SparseMatrixBase<Derived>& Amat, const float column_threshold) {
  const Derived& _A = Amat.derived();
  using Triplet     = Eigen::Triplet<Scalar>;
  using TripletVec  = std::vector<Triplet>;

  TLOG("Filtering the columns in a matrix [" << _A.rows() << " x " << _A.cols() << "]");

  //////////////////////////////
  // note: SpMat is row-major //
  //////////////////////////////

  const SpMat At = _A.transpose();

  Vec onesRow   = Mat::Ones(At.rows(), 1);
  Vec CountCols = At * onesRow;

  // Filter columns with less than some column_threshold
  std::vector<Index> valid_cols;
  for (Index j = 0; j < CountCols.size(); ++j) {
    if (CountCols(j) >= column_threshold) {
      valid_cols.push_back(j);
    }
  }

  TLOG("Found " << valid_cols.size() << "(with the sum >=" << column_threshold << ")");

  using Triplet    = Eigen::Triplet<Scalar>;
  using TripletVec = std::vector<Triplet>;

  TripletVec Tvec;

  const Index valid_max_cols = valid_cols.size();
  const Index INTERVAL       = 1000;
  const Index MAX_PRINT      = valid_max_cols / INTERVAL;

  for (Index j = 0; j < valid_max_cols; ++j) {
    const Index k = valid_cols.at(j);  // row of At
    for (SpMat::InnerIterator it(At, k); it; ++it) {
      const Index i = it.col();  // col of At
      Tvec.push_back(Triplet(i, j, it.value()));
    }

    if ((j + 1) % INTERVAL == 0) {
      std::cerr << "\r" << std::left << std::setfill('.') << std::setw(30) << "Adding ";
      std::cerr << std::right << std::setfill(' ');
      std::cerr << std::setw(10) << ((j + 1) / INTERVAL) << " x 1k columns";
      std::cerr << " (total " << std::right << std::setfill(' ');
      std::cerr << std::setw(10) << (MAX_PRINT) << ")";
      std::cerr << "\r" << std::flush;
    }
  }
  std::cerr << std::endl;

  SpMat ret(At.cols(), valid_cols.size());
  ret.reserve(Tvec.size());
  ret.setFromTriplets(Tvec.begin(), Tvec.end());

  return std::make_tuple(ret, valid_cols);
}

template <typename Vec>
auto eigen_argsort_descending(const Vec& data) {
  using Index = typename Vec::Index;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::execution::par, std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data(lhs) > data(rhs); });
  // std::sort(std::begin(index), std::end(index),
  //           [&](Index lhs, Index rhs) { return data(lhs) > data(rhs); });
  return index;
}

template <typename Derived>
Mat row_score_degree(const Eigen::SparseMatrixBase<Derived>& _xx) {
  const Derived& xx = _xx.derived();
  return xx.unaryExpr([](const Scalar x) { return std::abs(x); }) * Mat::Ones(xx.cols(), 1);
}

template <typename Derived>
Mat row_score_sd(const Eigen::SparseMatrixBase<Derived>& _xx) {
  const Derived& xx = _xx.derived();

  Vec s1         = xx * Mat::Ones(xx.cols(), 1);
  Vec s2         = xx.cwiseProduct(xx) * Mat::Ones(xx.cols(), 1);
  const Scalar n = xx.cols();
  Vec ret        = s2 - s1.cwiseProduct(s1 / n);
  ret            = ret / std::max(n - 1.0, 1.0);
  ret            = ret.cwiseSqrt();

  return ret;
}

#endif
