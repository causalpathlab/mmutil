#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <random>

#include "utils/util.hh"

#ifdef EIGEN_USE_MKL_ALL
#include "mkl_vsl.h"
#endif

#ifndef SGD_REPRESENTATION_GAUSSIAN_HH_
#define SGD_REPRESENTATION_GAUSSIAN_HH_

//////////////////////////////////////////////////////////////////////
// Gaussian representation class to accumulate stochastic gradients //
//                                                                  //
// Stochastic gradients                                             //
//                                                                  //
// G1 = sum F[s] Eps[s] / Sd[s]                                     //
// G2 = 1/2 * sum F[s] (Eps[s] * Eps[s] - 1) / Var[s]               //
//////////////////////////////////////////////////////////////////////

template <typename Derived>
auto make_gaus_repr(const Eigen::MatrixBase<Derived> &y);

template <typename Matrix>
void clear_repr(sgd_repr_gaus_t<Matrix> &);

template <typename Matrix, typename Derived>
void update_repr(sgd_repr_gaus_t<Matrix> &repr,
                 const Eigen::MatrixBase<Derived> &F);

template <typename Matrix, typename Derived>
void update_mean(sgd_repr_gaus_t<Matrix> &repr,
                 const Eigen::MatrixBase<Derived> &M);

template <typename Matrix, typename Derived>
void update_var(sgd_repr_gaus_t<Matrix> &repr,
                const Eigen::MatrixBase<Derived> &V);

///////////////////////////////////////////
// random normal generation by intel MKL //
///////////////////////////////////////////

#ifdef EIGEN_USE_MKL_ALL
template <typename Matrix>
const Matrix &sample_repr(sgd_repr_gaus_t<Matrix> &repr,
                          VSLStreamStatePtr rstream);

template <typename Matrix>
const Matrix &sample_repr_zeromean(sgd_repr_gaus_t<Matrix> &repr,
                                   VSLStreamStatePtr rstream);

#else

template <typename Matrix, typename RNG>
const Matrix &sample_repr(sgd_repr_gaus_t<Matrix> &repr, RNG &);

template <typename Matrix, typename RNG>
const Matrix &sample_repr_zeromean(sgd_repr_gaus_t<Matrix> &repr, RNG &);

#endif

template <typename Matrix>
const Matrix &sample_repr(sgd_repr_gaus_t<Matrix> &repr);

template <typename Matrix>
const Matrix &sample_repr_zeromean(sgd_repr_gaus_t<Matrix> &repr);

template <typename Matrix>
const Matrix &get_sampled_repr(sgd_repr_gaus_t<Matrix> &repr);

/////////////////////
// implementations //
/////////////////////

template <typename Matrix>
struct sgd_repr_gaus_t {
    using data_t = Matrix;
    using scalar_t = typename Matrix::Scalar;
    using index_t = typename Matrix::Index;

    explicit sgd_repr_gaus_t(const index_t _n, const index_t _m)
        : n(_n)
        , m(_m)
    {
#ifdef EIGEN_USE_MKL_ALL
        static_assert(std::is_same<scalar_t, float>::value,
                      "Must use float when using Eigen with MKL");
#endif

        summarized = false;
        n_add_sgd = 0;
    }

    ~sgd_repr_gaus_t() {}

    const Matrix &get_grad_type1()
    {
        if (!summarized)
            summarize();
        return G1;
    }

    const Matrix &get_grad_type2()
    {
        if (!summarized)
            summarize();
        return G2;
    }

    const Matrix &get_mean() const { return Mean; }
    const Matrix &get_var() const { return Var; }

    void summarize()
    {
        if (n_add_sgd > 0) {
            G1 = (FepsSdCum - epsSdCum.cwiseProduct(Fcum / n_add_sgd)) /
                n_add_sgd;
            G2 = 0.5 *
                (Feps1VarCum - eps1VarCum.cwiseProduct(Fcum / n_add_sgd)) /
                n_add_sgd;
        }
        summarized = true;
        n_add_sgd = 0;
    }

    const index_t rows() const { return n; }
    const index_t cols() const { return m; }

    const index_t n;
    const index_t m;

    Matrix G1; // stoch gradient wrt mean
    Matrix G2; // stoch gradient wrt var
    Matrix Eta; // random eta = Mean + Eps * Sd
    Matrix Eps; // Eps ~ N(0,1)
    Matrix Mean; // current mean
    Matrix Var; // current var

    Matrix Fcum; // cumulation of F
    Matrix FepsSdCum; // F * eps / Sd
    Matrix Feps1VarCum; // F * (eps^2 - 1) / Var
    Matrix epsSdCum; // eps / Sd
    Matrix eps1VarCum; // (eps^2 - 1) / Var

    Matrix epsSd; // eps / Sd
    Matrix eps1Var; // (eps^2 - 1) / Var

    bool summarized;
    scalar_t n_add_sgd;

    // helper functors
    struct eps_sd_op_t {
        const scalar_t operator()(const scalar_t &eps,
                                  const scalar_t &var) const
        {
            return eps / std::sqrt(var_min + var);
        }
        static constexpr scalar_t var_min = 1e-16;
    } EpsSd_op;

    struct eps_1var_op_t {
        const scalar_t operator()(const scalar_t &eps,
                                  const scalar_t &var) const
        {
            return (eps * eps - one_val) / (var_min + var);
        }
        static constexpr scalar_t var_min = 1e-16;
        static constexpr scalar_t one_val = 1.0;
    } Eps1Var_op;
};

template <typename Derived>
auto
make_gaus_repr(const Eigen::MatrixBase<Derived> &y)
{
    sgd_repr_gaus_t<typename Derived::scalar_t> ret(y.rows(), y.cols());
    clear_repr(ret);
    return ret;
}

template <typename Matrix>
void
clear_repr(sgd_repr_gaus_t<Matrix> &repr)
{
    const auto n = repr.n;
    const auto m = repr.m;

    repr.G1.setZero(n, m);
    repr.G2.setZero(n, m);
    repr.Eta.setZero(n, m);
    repr.Eps.setZero(n, m);
    repr.Mean.setZero(n, m);
    repr.Var.setZero(n, m);
    repr.Fcum.setZero(n, m);
    repr.FepsSdCum.setZero(n, m);
    repr.Feps1VarCum.setZero(n, m);
    repr.epsSdCum.setZero(n, m);
    repr.eps1VarCum.setZero(n, m);
    repr.epsSd.setZero(n, m);
    repr.eps1Var.setZero(n, m);
}

#ifdef EIGEN_USE_MKL_ALL

template <typename Matrix>
const Matrix &
sample_repr(sgd_repr_gaus_t<Matrix> &repr, VSLStreamStatePtr rstream)
{
    static_assert(std::is_same<float, typename Matrix::scalar_t>::value,
                  "Only assume float for intel RNG");

    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rstream, repr.Eps.size(),
                  repr.Eps.data(), 0.0, 1.0);
    repr.Eta = repr.Mean + repr.Eps.cwiseProduct(repr.Var.cwiseSqrt());
    return repr.Eta;
}

template <typename Matrix>
const Matrix &
sample_repr_zeromean(sgd_repr_gaus_t<Matrix> &repr, VSLStreamStatePtr rstream)
{
    static_assert(std::is_same<float, typename Matrix::scalar_t>::value,
                  "Only assume float for intel RNG");

    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rstream, repr.Eps.size(),
                  repr.Eps.data(), 0.0, 1.0);

    repr.Eta = repr.Eps.cwiseProduct(repr.Var.cwiseSqrt());
    return repr.Eta;
}

#else

template <typename Matrix, typename RNG>
const Matrix &
sample_repr(sgd_repr_gaus_t<Matrix> &repr, RNG &rng)
{
    using scalar_t = typename Matrix::scalar_t;
    std::normal_distribution<scalar_t> rnorm(0., 1.);
    repr.Eps = repr.Eps.unaryExpr([&](const auto &x) { return rnorm(rng); });
    repr.Eta = repr.Mean + repr.Eps.cwiseProduct(repr.Var.cwiseSqrt());
    return repr.Eta;
}

template <typename Matrix, typename RNG>
const Matrix &
sample_repr_zeromean(sgd_repr_gaus_t<Matrix> &repr, RNG &rng)
{
    using scalar_t = typename Matrix::scalar_t;
    std::normal_distribution<scalar_t> rnorm(0., 1.);
    repr.Eps = repr.Eps.unaryExpr([&](const auto &x) { return rnorm(rng); });
    return repr.Eta;
}

#endif

template <typename Matrix>
const Matrix &
sample_repr(sgd_repr_gaus_t<Matrix> &repr)
{
    using scalar_t = typename Matrix::scalar_t;
    repr.Eps = repr.Eps.unaryExpr(
        [&](const auto &x) { return static_cast<scalar_t>(ZIGG.norm()); });
    repr.Eta = repr.Mean + repr.Eps.cwiseProduct(repr.Var.cwiseSqrt());
    return repr.Eta;
}

template <typename Matrix>
const Matrix &
sample_repr_zeromean(sgd_repr_gaus_t<Matrix> &repr)
{
    using scalar_t = typename Matrix::scalar_t;
    repr.Eps = repr.Eps.unaryExpr(
        [&](const auto &x) { return static_cast<scalar_t>(ZIGG.norm()); });
    return repr.Eta;
}

template <typename Matrix>
const Matrix &
get_sampled_repr(sgd_repr_gaus_t<Matrix> &repr)
{
    return repr.Eta;
}

///////////////////////////////////////////////////////////////////////
// Accumulate stats corrected by control variates                    //
//                                                                   //
// G1 = mean_s Eps[s] / Sd[s] * (F[s] - EF)                          //
//    = mean_s(Eps[s] / Sd[s] * F[s])                                //
//      - mean_s(F[s]) * mean_s(Eps[s] / Sd[s])                      //
//                                                                   //
// G2 = 0.5 * mean_s (Eps[s] * Eps[s] - 1) / Var[s] * (F[s] - EF)    //
//    = 0.5 * mean_s (Eps[s] * Eps[s] - 1) / Var[s] * F[s]           //
//      - 0.5 * mean_s(F[s]) * mean_s (Eps[s] * Eps[s] - 1) / Var[s] //
///////////////////////////////////////////////////////////////////////

template <typename Matrix, typename Derived>
void
update_repr(sgd_repr_gaus_t<Matrix> &repr, const Eigen::MatrixBase<Derived> &F)
{
    repr.epsSd = repr.Eps.binaryExpr(repr.Var, repr.EpsSd_op);
    repr.eps1Var = repr.Eps.binaryExpr(repr.Var, repr.Eps1Var_op);

    if (repr.n_add_sgd == 0) {
        repr.Fcum = F;
        repr.epsSdCum = repr.epsSd;
        repr.eps1VarCum = repr.eps1Var;
        repr.FepsSdCum = F.cwiseProduct(repr.epsSd);
        repr.Feps1VarCum = F.cwiseProduct(repr.eps1Var);
    } else {
        repr.Fcum += F;
        repr.epsSdCum += repr.epsSd;
        repr.eps1VarCum += repr.eps1Var;
        repr.FepsSdCum += F.cwiseProduct(repr.epsSd);
        repr.Feps1VarCum += F.cwiseProduct(repr.eps1Var);
    }
    repr.n_add_sgd++;
    repr.summarized = false;
}

template <typename Matrix, typename Derived>
void
update_mean(sgd_repr_gaus_t<Matrix> &repr, const Eigen::MatrixBase<Derived> &M)
{
    repr.Mean = M.eval();
    repr.summarized = false;
}

template <typename Matrix, typename Derived>
void
update_var(sgd_repr_gaus_t<Matrix> &repr, const Eigen::MatrixBase<Derived> &V)
{
    repr.Var = V.eval();
    repr.summarized = false;
}

#endif
