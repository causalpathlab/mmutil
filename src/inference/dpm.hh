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
    using Index = typename T::Index;

    struct dpm_alpha_t : public check_positive_t<Scalar> {
        explicit dpm_alpha_t(const Scalar v)
            : check_positive_t<Scalar>(v)
        {
        }
    };

    struct num_clust_t : public check_positive_t<Index> {
        explicit num_clust_t(const Index v)
            : check_positive_t<Index>(v)
        {
        }
    };

    explicit trunc_dpm_t(const dpm_alpha_t dpm_alpha,
                         const num_clust_t num_clust)
        : a0(dpm_alpha.val)
        , K(num_clust.val)
        , logPr(K, 1)
        , u(K, 1)
        , v(K, 1)
        , sizeVec(K, 1)
        , sortedIndexes(K)
    {
        std::iota(sortedIndexes.begin(), sortedIndexes.end(), 0);
        logPr.setZero();
        u.setOnes();
        v.setOnes();
        sizeVec.setOnes();
    }

    void clear()
    {
        std::iota(sortedIndexes.begin(), sortedIndexes.end(), 0);
        logPr.setZero();
        u.setOnes();
        v.setOnes();
        sizeVec.setOnes();
    }

    template <typename Derived>
    void operator+=(const Eigen::MatrixBase<Derived> &Z)
    {
        _add(Z);
    }

    template <typename Derived>
    void operator-=(const Eigen::MatrixBase<Derived> &Z)
    {
        _subtract(Z);
    }

    template <typename Derived>
    void update(const Eigen::MatrixBase<Derived> &Z, const Scalar rate)
    {
#ifdef DEBUG
        ASSERT(Z.rows() == K, "Must feed K x n Z matrix");
#endif
        const Index n = Z.cols();

        if (n > 1) {
            if (onesN.rows() != n) {
                onesN.setConstant(n, 1, 1.0);
            }
            sizeVec *= (1.0 - rate);
            sizeVec += Z * onesN * rate;
        } else if (n == 1) {
            sizeVec *= (1.0 - rate);
            sizeVec += Z * rate;
        }

        posterior_update();
    }

    void add_to(const Index k)
    {
#ifdef DEBUG
        ASSERT(k >= 0 && k < K, "k must be in [0, " << K << ")");
#endif
        sizeVec(k) += 1.0;
        posterior_update();
    }

    void subtract_from(const Index k)
    {
#ifdef DEBUG
        ASSERT(k >= 0 && k < K, "k must be in [0, " << K << ")");
#endif
        sizeVec(k) -= 1.0;
        posterior_update();
    }

    void transfer(const Index k_old, const Index k_new)
    {
#ifdef DEBUG
        ASSERT(k_old >= 0 && k_old < K, "k_old must be in [0, " << K << ")");
        ASSERT(k_new >= 0 && k_new < K, "k_new must be in [0, " << K << ")");
#endif
        sizeVec(k_old) -= 1.0;
        sizeVec(k_new) += 1.0;
        posterior_update();
    }

private:
    template <typename Derived>
    void _add(const Eigen::MatrixBase<Derived> &Z)
    {
#ifdef DEBUG
        ASSERT(Z.rows() == K, "Must feed K x n Z matrix");
#endif
        const Index n = Z.cols();

        if (n > 1) {
            if (onesN.rows() != n) {
                onesN.resize(n, 1);
                onesN.setConstant(n, 1, 1.0);
            }
            sizeVec += Z * onesN;
        } else if (n == 1) {
            sizeVec += Z;
        }

        posterior_update();
    }

    template <typename Derived>
    void _subtract(const Eigen::MatrixBase<Derived> &Z)
    {
#ifdef DEBUG
        ASSERT(Z.rows() == K, "Must feed K x n Z matrix");
#endif
        const Index n = Z.cols();

        if (n > 1) {
            if (onesN.rows() != n) {
                onesN.resize(n, 1);
                onesN.setConstant(n, 1, 1.0);
            }
            sizeVec -= Z * onesN;
        } else if (n == 1) {
            sizeVec -= Z;
        }

        posterior_update();
    }

public:
    ///////////////////////////////////////////////////
    // u(k) = 1 + sum Z(:,k) = 1 + dpmstat(k)	   //
    // v(k) = a + sum_{l=k+1} u(k) = a + sum_{l=k+1} //
    ///////////////////////////////////////////////////
    void posterior_update()
    {
        auto comparator = [this](Index lhs, Index rhs) {
            return sizeVec(rhs) < sizeVec(lhs);
        };
        std::sort(sortedIndexes.begin(), sortedIndexes.end(), comparator);

        Scalar ntot = sizeVec.sum();
        Scalar cumsum = 0.0;

        for (auto k : sortedIndexes) {
            Scalar nk = sizeVec(k);
            cumsum += nk;
            u(k) = (1.0 + nk);
            v(k) = (a0 + ntot - cumsum);
        }
    }

    //////////////////////////////////////////////////////////////
    // log V(k) = digamma(u(k)) - digamma(u(k) + v(k))	      //
    // log (1- V(k)) = digamma(v(k)) - digamma(u(k) + v(k))     //
    // 							      //
    // log pi(k) = log V(k) + sum from l to (k-1) log(1 - V(l)) //
    //////////////////////////////////////////////////////////////

    const T &elbo()
    {
        auto comparator = [this](Index lhs, Index rhs) {
            return u(rhs) < u(lhs);
        };
        std::sort(sortedIndexes.begin(), sortedIndexes.end(), comparator);

        Scalar cum = 0.0;
        for (auto k : sortedIndexes) {
            Scalar log_denom = fasterdigamma(u(k) + v(k));
            logPr(k) = fasterdigamma(u(k)) - log_denom + cum;
            cum += std::log(v(k)) - log_denom;
        }

        return logPr;
    }

    const T &log_lcvi() { return elbo(); }

    const Scalar a0;
    const Index K;

private:
    T logPr;
    T u;
    T v;
    T sizeVec;
    T onesN;

    std::vector<Index> sortedIndexes;
};

#endif
