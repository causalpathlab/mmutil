#include <memory>

#include "inference/adam.hh"
#include "utils/check.hh"

#ifndef PARAMETERS_HH_
#define PARAMETERS_HH_

template <typename T>
struct param_traits {
    typedef typename T::data_t Matrix;
    typedef typename T::scalar_t Scalar;
    typedef typename T::index_t Index;
    typedef typename T::grad_adam_t Adam;
    typedef typename T::sgd_tag Sgd;
    typedef typename T::sparsity_tag Sparsity;
};

template <typename T>
using sgd_tag = typename param_traits<T>::Sgd;

template <typename T>
using sparsity_tag = typename param_traits<T>::Sparsity;

// q(theta[j,g]) ~ N(beta[j,g], 1/gamma[j,g])
struct param_tag_slab {
};

// q(theta[j,g]) ~ alpha[j,g] * N(beta[j,g], 1/gamma[j,g]) + (1-alpha[j,g])
// * delta(theta)
struct param_tag_spike_slab {
};

// q(theta[j,g]) ~ alpha[j,g] * gamma(mu[j,g], 1) + (1-alpha[j,g])
// * delta(theta)
struct param_tag_spike_gamma {
};

// q(theta[j,g]) ~ alpha[g] * gamma(mu[j,g], 1) + (1-alpha[g])
// * delta(theta)
struct param_tag_col_spike_gamma {
};

// q(theta[j,g]) ~ N(beta[j,g], 1/gamma[g])
struct param_tag_col_slab {
};

// q(theta[j,g]) ~ N(0, 1/gamma[g])
struct param_tag_col_slab_zero {
};

// q(theta[j,g]) ~ alpha[g] * N(beta[j,g], 1/gamma[g]) + (1-alpha[g]) *
// delta(theta)
struct param_tag_col_spike_slab {
};

// q(theta[j,g]) ~ alpha[j] * N(beta[j,g], 1/gamma[j]) + (1-alpha[j]) *
// delta(theta)
struct param_tag_row_spike_slab {
};

// q(theta[j,g]) ~ alpha[j,g] * N(beta[j,g], w[g] + w0[g])
//                (1-alpha[j,g]) * N(0, w0[g])
struct param_tag_mixture {
};

// theta ~ Beta(mu * phi, (1 - mu) * phi)
struct param_tag_beta {
};

struct param_tag_sparse {
};
struct param_tag_dense {
};

////////////////////////////////////////////////////////////////
// include implementations
#include "inference/param_beta.hh"
#include "inference/param_col_slab.hh"
#include "inference/param_col_slab_zero.hh"
#include "inference/param_col_spike_gamma.hh"
#include "inference/param_col_spike_slab.hh"
#include "inference/param_mixture.hh"
#include "inference/param_row_spike_slab.hh"
#include "inference/param_slab.hh"
#include "inference/param_spike_gamma.hh"
#include "inference/param_spike_slab.hh"

template <typename Derived>
using SparseDeriv = Eigen::SparseMatrixBase<Derived>;

template <typename Derived>
using DenseDeriv = Eigen::MatrixBase<Derived>;

////////////////////////////////////////////////////////////////
// dispatch functions for SGD evaluation
template <typename P, typename D1, typename D2, typename D3>
void
param_eval_sgd(P &p, const Eigen::MatrixBase<D1> &g1, //
               const Eigen::MatrixBase<D2> &g2, //
               const Eigen::MatrixBase<D3> &nobs)
{
    param_safe_eval_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

template <typename P, typename D1, typename D2, typename D3>
void
param_eval_sgd(P &p, const Eigen::SparseMatrixBase<D1> &g1, //
               const Eigen::SparseMatrixBase<D2> &g2, //
               const Eigen::SparseMatrixBase<D3> &nobs)
{
    param_safe_eval_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

// check to match sparsity of the parameter and gradient
template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void
param_safe_eval_sgd(Parameter &P, const Eigen::MatrixBase<Deriv1> &G1, //
                    const Eigen::MatrixBase<Deriv2> &G2, //
                    const Eigen::MatrixBase<Deriv3> &Nobs, //
                    const param_tag_dense)
{
    param_impl_eval_sgd(P, G1.derived(), G2.derived(), Nobs.derived(),
                        sgd_tag<Parameter>());
}

template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void
param_safe_eval_sgd(Parameter &P,
                    const Eigen::SparseMatrixBase<Deriv1> &G1, //
                    const Eigen::SparseMatrixBase<Deriv2> &G2, //
                    const Eigen::SparseMatrixBase<Deriv3> &Nobs, //
                    const param_tag_sparse)
{
    param_impl_eval_sgd(P, G1.derived(), G2.derived(), Nobs.derived(),
                        sgd_tag<Parameter>());
}

////////////////////////////////////////////////////////////////
// dispatch functions for SGD evaluation
template <typename P, typename D1, typename D2, typename D3>
void
hyperparam_eval_sgd(P &p, const Eigen::MatrixBase<D1> &g1, //
                    const Eigen::MatrixBase<D2> &g2, //
                    const Eigen::MatrixBase<D3> &nobs)
{
    hyperparam_safe_eval_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

template <typename P, typename D1, typename D2, typename D3>
void
hyperparam_eval_sgd(P &p, const Eigen::SparseMatrixBase<D1> &g1, //
                    const Eigen::SparseMatrixBase<D2> &g2, //
                    const Eigen::SparseMatrixBase<D3> &nobs)
{
    hyperparam_safe_eval_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

// check to match sparsity of the parameter and gradient
template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void
hyperparam_safe_eval_sgd(Parameter &P,
                         const Eigen::MatrixBase<Deriv1> &G1, //
                         const Eigen::MatrixBase<Deriv2> &G2, //
                         const Eigen::MatrixBase<Deriv3> &Nobs,
                         const param_tag_dense)
{
    hyperparam_impl_eval_sgd(P, G1.derived(), G2.derived(), Nobs.derived(),
                             sgd_tag<Parameter>());
}

template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void
hyperparam_safe_eval_sgd(Parameter &P,
                         const Eigen::SparseMatrixBase<Deriv1> &G1, //
                         const Eigen::SparseMatrixBase<Deriv2> &G2, //
                         const Eigen::SparseMatrixBase<Deriv3> &Nobs, //
                         const param_tag_sparse)
{
    hyperparam_impl_eval_sgd(P, G1.derived(), G2.derived(), Nobs.derived(),
                             sgd_tag<Parameter>());
}

////////////////////////////////////////////////////////////////
// dispatch functions for initialization
template <typename Parameter>
void
param_initialize(Parameter &P)
{
    param_impl_initialize(P, sgd_tag<Parameter>());
}

// dispatch functions for update
template <typename Parameter, typename Scalar>
void
param_update_sgd(Parameter &P, const Scalar rate)
{
    param_impl_update_sgd(P, rate, sgd_tag<Parameter>());
}

template <typename Parameter, typename Scalar>
void
hyperparam_update_sgd(Parameter &P, const Scalar rate)
{
    hyperparam_impl_update_sgd(P, rate, sgd_tag<Parameter>());
}

// dispatch functions for resolution
template <typename Parameter>
void
param_resolve(Parameter &P)
{
    param_impl_resolve(P, sgd_tag<Parameter>());
}

// dispatch functions for resolution
template <typename Parameter>
void
hyperparam_resolve(Parameter &P)
{
    hyperparam_impl_resolve(P, sgd_tag<Parameter>());
}

// dispatch functions for perturbation
template <typename Parameter, typename Scalar>
void
param_perturb(Parameter &P, const Scalar sd)
{
    param_impl_perturb(P, sd, sgd_tag<Parameter>());
}

template <typename Parameter, typename Scalar, typename RNG>
void
param_perturb(Parameter &P, const Scalar sd, RNG &rng)
{
    param_impl_perturb(P, sd, rng, sgd_tag<Parameter>());
}

template <typename Parameter>
void
param_check_nan(Parameter &P, std::string msg)
{
    std::cerr << msg;
    param_impl_check_nan(P, sgd_tag<Parameter>());
    std::cerr << " -> ok" << std::endl;
}

template <typename Parameter>
auto
param_log_odds(Parameter &P)
{
    return param_impl_log_odds(P, sgd_tag<Parameter>());
}

template <typename Parameter>
const auto &
param_mean(Parameter &P)
{
    return param_impl_mean(P, sgd_tag<Parameter>());
}

template <typename Parameter>
const auto &
param_var(Parameter &P)
{
    return param_impl_var(P, sgd_tag<Parameter>());
}

template <typename Parameter>
void
param_write(Parameter &P, const std::string hdr, const std::string gz)
{
    param_impl_write(P, hdr, gz, sgd_tag<Parameter>());
}

#endif
