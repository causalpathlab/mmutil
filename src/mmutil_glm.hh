#include "mmutil.hh"

#ifndef MMUTIL_GLM_HH_
#define MMUTIL_GLM_HH_

struct poisson_pseudo_response_t {
    /// @param y
    /// @param eta
    Scalar operator()(const Scalar &y, const Scalar &eta) const
    {
        return eta + fasterexp(-eta) * y - one;
    }
    static constexpr Scalar one = 1.0;
};

struct poisson_weight_t {
    /// @param eta
    Scalar operator()(const Scalar &eta) const { return fasterexp(eta); }
    static constexpr Scalar one = 1.0;
};

struct poisson_llik_t {
    Scalar operator()(const Scalar &y, const Scalar &eta) const
    {
        return y * eta - fasterexp(eta);
    }
};

/// Fit GLM by coordinate-wise descent:
/// fit.glm <- function(xx, y, reg = 1) {
///   p = ncol(xx)
///   beta = rep(0, p)
///   beta.se = rep(sqrt(reg), p)
///   eta = xx %*% beta
///   llik = 0
///   for(iter in 1:100) {
///     llik.old = llik
///     y.pseudo = -1 + eta + exp(-eta) * y
///     ww = exp(eta)
///     for(j in 1:p) {
///       .r = eta - xx[, j] * beta[j]
///       .num = sum(ww * xx[, j] * (y.pseudo - .r))
///       .denom = sum(ww * xx[, j]^2) + reg
///       beta[j] = .num / .denom
///       beta.se[j] = 1 / sqrt(sum(ww * xx[, j]^2))
///       eta = .r + xx[, j] * beta[j]
///     }
///     llik = mean(y * eta - exp(eta))
///     if(abs(llik.old - llik) / abs(llik.old + 1e-4) < 1e-4){
///       break
///     }
///   }
///   list(beta = beta, se = beta.se)
/// }
template <typename pseudo_response_t, typename weight_t>
void
fit_glm(Mat xx, Mat y, const Index max_iter, const Scalar reg)
{
    pseudo_response_t resp;
    weight_t weight;

    const Index n = xx.nrow();
    const Index p = xx.ncol();

    Mat _beta = Mat::Zero(p, 1);
    Mat _eta = xx * beta;
    Mat _y(n, 1);
    Mat _w(n, 1);
    Mat _r(n, 1);

    for (Index iter = 0; iter < max_iter; ++iter) {

        _y = y.binaryExpr(_eta, resp);
        _w = _eta.unaryExpr(weight);

        for (Index j = 0; j < p; ++j) {
            _r = _eta - xx.col(j) * _beta(j);

            const Scalar num =
                (_y - _r).cwiseProduct(xx.col(j)).cwiseProduct(_w).sum();

            const Scalar denom =
                xx.col(j).cwiseProduct(xx.col(j)).cwiseProduct(_w).sum();

            _beta(j) = num / (denom + reg);

            _eta = _r + xx.col(j) * _beta(j);
        }
    }
}

#endif
