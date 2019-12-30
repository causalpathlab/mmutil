////////////////////////////////////////////////////////
// truncated dirichlet process prior for all k	      //
// 						      //
// ln pi(k) = ln u(k) - ln [u(k) + v(k)]	      //
//            + sum[1, K) [ln u(l) - ln(u(l) + v(l))] //
// 						      //
// where					      //
// 						      //
// u(k) = 1 + sum Z(:,k) = 1 + dpmstat(k)	      //
// v(k) = a + sum_{l=k+1} u(k) = a + sum_{l=k+1}      //
////////////////////////////////////////////////////////

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>

#include "utils/check.hh"
#include "utils/util.hh"

#ifndef DPM_HH_
#define DPM_HH_

template <typename T>
struct trunc_dpm_t {
  using Scalar = typename T::Scalar;
  using Index  = typename T::Index;
  using Dense  = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  struct dpm_alpha_t : public check_positive_t<Scalar> {
    explicit dpm_alpha_t(const Scalar v) : check_positive_t<Scalar>(v) {}
  };

  struct num_clust_t : public check_positive_t<Index> {
    explicit num_clust_t(const Index v) : check_positive_t<Index>(v) {}
  };

  explicit trunc_dpm_t(const dpm_alpha_t& dpm_alpha, const num_clust_t& num_clust)
      : a0(dpm_alpha.val),
        K(num_clust.val),
        logPr(K, 1),
        u(K, 1),
        v(K, 1),
        sizeVec(K, 1),
        sortedIndexes(K) {
    std::iota(sortedIndexes.begin(), sortedIndexes.end(), 0);
    logPr.setZero();
    u.setOnes();
    v.setOnes();
    sizeVec.setOnes();
  }

  template <typename Derived>
  void update(const Eigen::MatrixBase<Derived>& Z, const Scalar rate) {
#ifdef DEBUG
    ASSERT(Z.rows() == K, "Must feed K x n Z matrix");
#endif
    const auto n = Z.cols();
    if (n > 1) {
      if (onesN.rows() != n) {
        onesN.setConstant(n, 1, 1.0);
      }
      sizeVec = sizeVec * (1.0 - rate) + Z * onesN * rate;
    } else {
      sizeVec = sizeVec * (1.0 - rate) + Z * rate;
    }

    _update(rate);
  }

  ////////////////////////////////////////////////////////////////
  // ln pi(k) = ln u(k) - ln [u(k) + v(k)]
  //            + sum[1, K) [ln u(l) - ln(u(l) + v(l))]
  const T& eval() {
    auto comparator = [this](Index lhs, Index rhs) { return u(rhs) < u(lhs); };
    std::sort(sortedIndexes.begin(), sortedIndexes.end(), comparator);

    Scalar cum = 0.0;
    for (auto k : sortedIndexes) {
      Scalar log_denom = std::log(u(k) + v(k));
      logPr(k)         = std::log(u(k)) - log_denom + cum;
      cum += std::log(v(k)) - log_denom;
    }

    return logPr;
  }

  const Scalar a0;
  const Index K;

 private:
  T logPr;
  T u;
  T v;
  T sizeVec;
  Dense onesN;

  ////////////////////////////////////////////////////////////////
  // u(k) = 1 + sum Z(:,k) = 1 + dpmstat(k)
  // v(k) = a + sum_{l=k+1} u(k) = a + sum_{l=k+1}
  void _update(const Scalar rate) {
    auto comparator = [this](Index lhs, Index rhs) { return sizeVec(rhs) < sizeVec(lhs); };
    std::sort(sortedIndexes.begin(), sortedIndexes.end(), comparator);

    Scalar ntot   = sizeVec.sum();
    Scalar cumsum = 0.0;

    for (auto k : sortedIndexes) {
      Scalar nk = sizeVec(k);
      cumsum += nk;

      u(k) = (1.0 - rate) * u(k) + rate * (1.0 + nk);
      v(k) = (1.0 - rate) * v(k) + rate * (a0 + ntot - cumsum);
    }
  }

  std::vector<Index> sortedIndexes;
};

#endif
