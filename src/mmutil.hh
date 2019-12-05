#include <iostream>
#include "utils/io.hh"
#include "utils/util.hh"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/lexical_cast.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>

#ifndef MMUTIL_HH_
#define MMUTIL_HH_

template <typename Derived>
auto find_independent_components(Eigen::SparseMatrixBase<Derived>& Smat,
                                 const float sn_cutoff) {
  // Construct boost graph
  Derived& S = Smat.derived();

  using Scalar = typename Derived::Scalar;
  using SpMat = typename Eigen::SparseMatrix<Scalar>;
  using Index = typename SpMat::Index;

  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            boost::no_property, boost::no_property>;

  using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
  using Edge = typename Graph::edge_descriptor;

  TLOG("Building an undirected and unweighted graph");

  Graph G;
  const Index nrow = S.rows();
  const Index max_vertex = nrow + S.cols();

  for (Index u = boost::num_vertices(G); u <= max_vertex; ++u)
    boost::add_vertex(G);

  for (Index j = 0; j < S.outerSize(); ++j) {
    for (typename SpMat::InnerIterator it(S, j); it; ++it) {
      bool has_edge;
      Edge e;

      if (it.value() >= sn_cutoff) {
        const auto u = it.row();
        const auto v = it.col() + nrow;

        boost::tie(e, has_edge) = boost::edge(u, v, G);
        if (!has_edge) boost::add_edge(u, v, G);
      }
    }
  }

  TLOG("Checking connected components in the shared-neighborhood graph");

  using IndexVec = std::vector<Index>;
  IndexVec membership(boost::num_vertices(G));
  const Index numComp = boost::connected_components(G, &membership[0]);

  TLOG("Found " << numComp << " connected components");

  return membership;
}

template <typename Derived>
auto find_independent_columns(Eigen::SparseMatrixBase<Derived>& Amat,
                              const float sn_cutoff) {
  using Scalar = typename Derived::Scalar;
  using SpMat = typename Eigen::SparseMatrix<Scalar>;
  using Mat = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  Derived& A = Amat.derived();
  SpMat Adj = A;
  Adj.coeffs() /= Adj.coeffs();

  Mat AdjDense = Mat(Adj);
  Mat Sdense = AdjDense.transpose() * AdjDense;

  TLOG("Take a dense Shared Neighbor Matrix");

  SpMat S = Sdense.sparseView();

  TLOG("Back to the sparse matrix");

  return find_independent_components(S, sn_cutoff);
}

template <typename Derived>
auto find_independent_rows(Eigen::SparseMatrixBase<Derived>& Amat,
                           const float sn_cutoff) {
  using Scalar = typename Derived::Scalar;
  using SpMat = typename Eigen::SparseMatrix<Scalar>;

  Derived& A = Amat.derived();
  SpMat Adj = A;
  Adj.coeffs() /= Adj.coeffs();

  SpMat S = Adj * Adj.transpose();
  return find_independent_components(S, sn_cutoff);
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

  TLOG("Found " << valid_cols.size() << " columns (with the sum >=" << column_threshold
                << ")");

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
