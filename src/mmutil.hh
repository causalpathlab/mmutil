#include <iostream>
#include "utils/io.hh"
#include "utils/util.hh"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/lexical_cast.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#ifndef MMUTIL_HH_
#define MMUTIL_HH_

template <typename EigenVec>
inline auto std_vector(const EigenVec& eigen_vec) {
  std::vector<typename EigenVec::Scalar> ret(eigen_vec.size());
  for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j)
    ret[j] = eigen_vec(j);
  return ret;
}

template <typename TVEC>
inline auto build_eigen_triplets(const TVEC& Tvec) {
  using Scalar = float;
  using SpMat = Eigen::SparseMatrix<Scalar>;
  using Index = SpMat::Index;

  using _Triplet = Eigen::Triplet<Scalar>;
  using _TripletVec = std::vector<_Triplet>;
  _TripletVec ret;

  for (auto tt : Tvec) {
    Index r;
    Index c;
    Scalar w;
    std::tie(r, c, w) = tt;

    ret.push_back(_Triplet(r, c, w));
  }
  return ret;
}

template <typename TVEC, typename INDEX>
inline auto build_eigen_sparse(const TVEC& Tvec, const INDEX max_row,
                               const INDEX max_col) {
  using Scalar = float;
  using SpMat = Eigen::SparseMatrix<Scalar>;
  using Index = SpMat::Index;

  auto _tvec = build_eigen_triplets(Tvec);

  SpMat ret(max_row, max_col);
  ret.setFromTriplets(_tvec.begin(), _tvec.end());
  return ret;
}

template <typename Derived>
auto filter_columns(Eigen::SparseMatrixBase<Derived>& Amat,
                    const float column_threshold) {
  Derived& A = Amat.derived();
  using Scalar = typename Derived::Scalar;
  using SpMat = typename Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
  using Triplet = Eigen::Triplet<Scalar>;
  using TripletVec = std::vector<Triplet>;
  using Index = typename Eigen::SparseMatrix<Scalar>::Index;

  using Mat = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  Vec onesRow = Mat::Ones(A.rows(), 1);
  Vec CountCols = A.transpose() * onesRow;

  // Filter columns with less than some column_threshold
  std::vector<Scalar> count_cols(CountCols.size());
  Vec::Map(&count_cols[0], CountCols.size()) = CountCols;
  std::vector<Index> valid_cols;

  for (Index j = 0; j < count_cols.size(); ++j) {
    if (count_cols.at(j) >= column_threshold) {
      valid_cols.push_back(j);
    }
  }

  TLOG("Found " << valid_cols.size()
                << " columns (with the sum >=" << column_threshold << ")");

  using Triplet = Eigen::Triplet<Scalar>;
  using TripletVec = std::vector<Triplet>;

  TripletVec Tvec;

  for (Index j = 0; j < valid_cols.size(); ++j) {
    const Index k = valid_cols.at(j);
    for (typename SpMat::InnerIterator it(A, k); it; ++it) {
      Tvec.push_back(Triplet(it.row(), j, it.value()));
    }
  }

  SpMat ret(A.rows(), valid_cols.size());
  ret.setFromTriplets(Tvec.begin(), Tvec.end());

  return std::make_tuple(ret, valid_cols);
}

#endif
