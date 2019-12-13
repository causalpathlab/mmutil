#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/SVD>

#ifndef _SVD_HH_
#define _SVD_HH_

////////////////////
// randomized SVD //
////////////////////

// Implement Alg 4.4 of Halko et al. (2009)
// Modified from https://github.com/kazuotani14/RandomizedSvd

template <typename T>
class RandomizedSVD {
 public:
  RandomizedSVD(const int _max_rank, const int _iter)
      : max_rank(_max_rank), iter(_iter), U(), D(), V(), qq() {}

  using Vec = Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, 1>;

  const T matrixU() const { return U; }
  const T matrixV() const { return V; }
  const Vec singularValues() const { return D; }

  template <typename Derived>
  void compute(Eigen::MatrixBase<Derived> const& X) {
    using Index    = typename Derived::Index;
    const Index nr = X.rows();
    const Index nc = X.cols();

    int rank       = std::min(nr, nc);
    int oversample = 0;

    if (max_rank > 0 && max_rank < rank) {
      rank       = max_rank;
      oversample = 5;
    }

    ASSERT(rank > 0, "Must be at least rank = 1");

    qq.resize(nr, rank);
    U.resize(nr, rank);
    V.resize(nc, rank);
    D.resize(rank, 1);

    qq.setZero();
    rand_subspace_iteration(X, rank + oversample);
    qq = qq.leftCols(rank);

    T B = qq.transpose() * X;

    if (verbose) TLOG("Final svd on [" << B.rows() << "x" << B.cols() << "]");

    // Eigen::JacobiSVD<T> svd;
    Eigen::BDCSVD<T> svd;
    svd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (verbose) TLOG("Construct U, D, V");

    U = (qq * svd.matrixU()).block(0, 0, nr, rank);
    V = svd.matrixV().block(0, 0, nc, rank);
    D = svd.singularValues().head(rank);
  }

  void set_verbose() { verbose = true; }

  const int max_rank;
  const int iter;

 private:
  T U;
  Vec D;
  T V;
  T qq;
  bool verbose;

  template <typename Derived>
  void rand_subspace_iteration(Eigen::MatrixBase<Derived> const& X, const int size) {
    using Index    = typename Derived::Index;
    const Index nr = X.rows();
    const Index nc = X.cols();

    T L(nr, size);
    T Q = T::Random(nc, size);

    // Use LU normalization since QR is too slow in Eigen

    Eigen::FullPivLU<T> lu1(nr, size);
    Eigen::FullPivLU<T> lu2(nc, nr);

    if (verbose) TLOG("Find Q in randomized svd...");

    for (int i = 0; i < iter; ++i) {
      if (verbose) TLOG("Start : LU iteration " << (i + 1));

      lu1.compute(X * Q);
      L.setIdentity();
      L.block(0, 0, nr, size).template triangularView<Eigen::StrictlyLower>() = lu1.matrixLU();

      lu2.compute(X.transpose() * L);
      Q.setIdentity();
      Q.block(0, 0, nc, size).template triangularView<Eigen::StrictlyLower>() = lu2.matrixLU();

      if (verbose) TLOG("Done : LU iteration " << (i + 1));
    }

    Eigen::ColPivHouseholderQR<T> qr(X * Q);
    qq = qr.householderQ() * T::Identity(nr, size);
    if (verbose) TLOG("Found Q");
  }
};

#endif
