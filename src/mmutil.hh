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

using Scalar = float;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
using Index = SpMat::Index;

using Mat = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename EigenVec>
inline auto std_vector(const EigenVec& eigen_vec) {
  std::vector<typename EigenVec::Scalar> ret(eigen_vec.size());
  for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) ret[j] = eigen_vec(j);
  return ret;
}

template <typename TVEC>
inline auto build_eigen_triplets(const TVEC& Tvec) {
  using _Triplet = Eigen::Triplet<Scalar>;
  using _TripletVec = std::vector<_Triplet>;
  _TripletVec ret;

  const Index INTERVAL = 1e6;
  const Index max_tvec_size = Tvec.size();
  Index j = 0;

  for (auto tt : Tvec) {
    Index r;
    Index c;
    Scalar w;
    std::tie(r, c, w) = tt;

    ret.push_back(_Triplet(r, c, w));

    if (++j % INTERVAL == 0) {
      std::cerr << "\r" << std::setw(30) << "Adding " << std::setw(10) << (j / INTERVAL)
                << " x 1M triplets (total " << std::setw(10) << (max_tvec_size / INTERVAL) << ")"
                << std::flush;
    }
  }
  std::cerr << std::endl;
  return ret;
}

template <typename TVEC, typename INDEX>
inline auto build_eigen_sparse(const TVEC& Tvec, const INDEX max_row, const INDEX max_col) {
  auto _tvec = build_eigen_triplets(Tvec);

  SpMat ret(max_row, max_col);
  ret.setFromTriplets(_tvec.begin(), _tvec.end());
  return ret;
}

template <typename Derived>
auto filter_columns(const Eigen::SparseMatrixBase<Derived>& Amat, const float column_threshold) {
  const Derived& _A = Amat.derived();
  using Triplet = Eigen::Triplet<Scalar>;
  using TripletVec = std::vector<Triplet>;

  TLOG("Filtering the columns matrix [" << _A.rows() << " x " << _A.cols() << "]");

  //////////////////////////////
  // note: SpMat is row-major //
  //////////////////////////////

  const SpMat At = _A.transpose();

  Vec onesRow = Mat::Ones(At.rows(), 1);
  Vec CountCols = At * onesRow;

  // Filter columns with less than some column_threshold
  std::vector<Index> valid_cols;
  for (Index j = 0; j < CountCols.size(); ++j) {
    if (CountCols(j) >= column_threshold) {
      valid_cols.push_back(j);
    }
  }

  TLOG("Found " << valid_cols.size());
  TLOG("(with the sum >=" << column_threshold << ")");

  using Triplet = Eigen::Triplet<Scalar>;
  using TripletVec = std::vector<Triplet>;

  TripletVec Tvec;

  const Index valid_max_cols = valid_cols.size();
  const Index INTERVAL = 1000;

  for (Index j = 0; j < valid_max_cols; ++j) {
    const Index k = valid_cols.at(j);  // row of At
    for (SpMat::InnerIterator it(At, k); it; ++it) {
      const Index i = it.col();  // col of At
      Tvec.push_back(Triplet(i, j, it.value()));
    }

    if ((j + 1) % INTERVAL == 0) {
      std::cerr << "\r" << std::setw(30) << "Adding " << std::setw(10) << (j / INTERVAL)
                << " x 1k columns  (total " << std::setw(10) << (valid_max_cols / INTERVAL) << ")"
                << std::flush;
    }
  }
  std::cerr << std::endl;

  SpMat ret(At.cols(), valid_cols.size());
  ret.setFromTriplets(Tvec.begin(), Tvec.end());

  return std::make_tuple(ret, valid_cols);
}

template <typename Vec>
auto eigen_argsort_descending(const Vec& data) {
  using Index = typename Vec::Index;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data(lhs) > data(rhs); });
  return index;
}

#endif
