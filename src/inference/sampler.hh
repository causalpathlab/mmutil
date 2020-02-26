#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <random>

#include "utils/fastexp.h"
#include "utils/fastlog.h"

#ifndef SAMPLER_HH_
#define SAMPLER_HH_

template <typename Scalar, typename Index>
struct discrete_sampler_t {
    explicit discrete_sampler_t(const Index k)
        : K(k)
    {
    }

    template <typename Derived>
    Index operator()(const Eigen::MatrixBase<Derived> &xx)
    {
        Index argmax_k;

        const Scalar maxval = xx.maxCoeff(&argmax_k);
        const Scalar exp_sumval = xx.unaryExpr([&maxval](const Scalar x) {
                                        return fasterexp(x - maxval);
                                    })
                                      .sum();

        const Scalar u = Unif(Rng) * exp_sumval;
        Scalar cum = 0.0;
        Index rIndex = 0;

        for (Index k = 0; k < K; ++k) {
            const Scalar val = xx(k);
            cum += fasterexp(val - maxval);
            if (u <= cum) {
                rIndex = k;
                break;
            }
        }
        return rIndex;
    }

    const Index K;

private:
    std::mt19937 Rng{ std::random_device{}() };
    std::uniform_real_distribution<Scalar> Unif{ 0.0, 1.0 };

    template <typename Derived>
    inline Scalar _log_sum_exp(const Eigen::MatrixBase<Derived> &log_vec)
    {
        const Derived &xx = log_vec.derived();

        Scalar maxlogval = xx(0);
        for (Index j = 1; j < xx.size(); ++j) {
            if (xx(j) > maxlogval)
                maxlogval = xx(j);
        }

        Scalar ret = 0;
        for (Index j = 0; j < xx.size(); ++j) {
            ret += fasterexp(xx(j) - maxlogval);
        }
        return fasterlog(ret) + maxlogval;
    }
};

#endif
