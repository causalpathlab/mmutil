#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "utils/check.hh"
#include "utils/fastexp.h"
#include "utils/fastgamma.h"
#include "utils/fastlog.h"
#include "utils/util.hh"

#ifndef COMPONENT_GAUSSIAN_HH_
#define COMPONENT_GAUSSIAN_HH_

/////////////////////////////////
// x   ~ N(mu, 1/tau * I)      //
// mu  ~ N(0, scale / tau * I) //
// tau ~ Gamma(a0, b0)         //
/////////////////////////////////

template <typename T>
struct multi_gaussian_component_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using mat_type =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using vec_type = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>;

    struct dim_t : public check_positive_t<Scalar> {
        explicit dim_t(const Index v)
            : check_positive_t<Scalar>(v)
        {
        }
    };

    explicit multi_gaussian_component_t(const dim_t _dim)
        : p(static_cast<Index>(_dim.val))
        , // dimensionality
        d(static_cast<Scalar>(p))
        , // dimensionality
        n(0.)
        , // sample size
        s1(p)
        , // 1st moment sum
        s2(0.)
        , // 2nd moment sum
        scale(1e-2)
        , // hyper for mean prior
        a0(1e-2)
        , // hyper for precision prior
        b0(100.)
        , // hyper for precision prior
        mu(p)
        , // variational (posterior) mean
        mu_prec(a0 / b0 / scale)
        , // prec of vari-mean
        tau(a0 / b0)
        , // precision
        lntau(0.)
        , // log precision
        musq(0.)
    { // mean square
        // constructor
        s1.setZero();
        mu.setZero();
        posterior_update();
    }

    void clear()
    {
        s1.setZero();
        s2 = 0;
        mu.setZero();
        mu_prec = a0 / b0 / scale;
        tau = a0 / b0;
        posterior_update();
    }

    /////////////////////
    // update routines //
    /////////////////////

    template <typename Derived>
    inline void update(const Eigen::MatrixBase<Derived> &xx, //
                       const Scalar z_old,                   //
                       const Scalar z_new)
    {
        const Derived &x = xx.derived(); // d x 1

#ifdef DEBUG
        ASSERT(x.rows() == s1.rows(), "x1.rows() != s1.rows()");
        ASSERT(z_old >= 0 && z_old <= 1, "z_old in [0, 1] : " << z_old);
        ASSERT(z_new >= 0 && z_new <= 1, "z_new in [0, 1] : " << z_new);
#endif

        const Scalar x2 = x.cwiseProduct(x).sum();

        n -= z_old;
        s1 -= x * z_old;
        s2 -= x2 * z_old;

#ifdef DEBUG
        ASSERT(n > (-1e-8), "Cannot have negative n");
#endif

        n += z_new;
        s1 += x * z_new;
        s2 += x2 * z_new;

#ifdef DEBUG
        ASSERT(n > (-1e-8), "Cannot have negative n");
#endif

        posterior_update();
    }

    template <typename Derived>
    inline multi_gaussian_component_t &
    operator+=(const Eigen::MatrixBase<Derived> &xx)
    {
        return _add(xx);
    }

    template <typename Derived>
    inline multi_gaussian_component_t &
    operator-=(const Eigen::MatrixBase<Derived> &xx)
    {
        return _subtract(xx);
    }

    ////////////
    // scores //
    ////////////

    /////////////////////////////
    // variational lower-bound //
    /////////////////////////////

    template <typename Derived>
    Scalar elbo(const Eigen::MatrixBase<Derived> &xx) const
    { // dim x 1

        const Derived &x = xx.derived();
#ifdef DEBUG
        ASSERT(x.cols() == 1, "only support dim x 1 in elbo()");
#endif

        const Scalar x2 = x.cwiseProduct(x).sum();
        const Scalar mux = mu.cwiseProduct(x).sum();

        Scalar ret = -0.5 * d * ln2pi;
        ret += 0.5 * d * lntau;
        ret -= 0.5 * tau * (x2 - 2.0 * mux + musq);
        return ret;
    }

    /////////////////////////////////////////////
    // locally collapsed variational inference //
    /////////////////////////////////////////////

    // log N(x|variational mean[x], variational var[mu] + variational
    // var[x])
    template <typename Derived>
    Scalar log_lcvi(const Eigen::MatrixBase<Derived> &xx) const
    { // dim x 1

        const Derived &x = xx.derived();
#ifdef DEBUG
        ASSERT(x.cols() == 1, "only support dim x 1 in log_lcvi()");
#endif

        const Scalar rss = (x - mu).cwiseProduct(x - mu).sum();
        const Scalar var = 1.0 / mu_prec + 1.0 / tau;

        Scalar ret = -rss / var * 0.5;
        ret -= 0.5 * d * fasterlog(var);
        ret -= 0.5 * d * ln2pi;
        return ret;
    }

    /////////////////////////
    // marginal likelihood //
    /////////////////////////

    Scalar log_marginal() const
    {
        // mu = s1 / (scale + n);  // posterior mean
        const Scalar C = s2 - (s1 / (scale + n)).cwiseProduct(s1).sum();

        Scalar ret = -0.5 * n * d * ln2pi;
        ret += 0.5 * d * fasterlog(scale);
        ret -= 0.5 * d * fasterlog(n + scale);
        ret += fasterlgamma(a0 + n * d * 0.5);
        ret -= fasterlgamma(a0);
        ret += a0 * fasterlog(b0);
        ret -= (a0 + n * d * 0.5) * fasterlog(b0 + C * 0.5);

        return ret;
    }

    template <typename Derived>
    Scalar log_marginal_ratio(const Eigen::MatrixBase<Derived> &xx) const
    {
        const Derived &x = xx.derived();
#ifdef DEBUG
        ASSERT(x.cols() == 1, "only support dim x 1 in log_lcvi()");
#endif

        // mu = (s1 + x) / (scale + n + 1);  // posterior mean
        // s2 = s2 + sum x^2
        const Scalar x2 = x.cwiseProduct(x).sum();

        const Scalar C_new =
            s2 + x2 - ((s1 + x) / (scale + n + 1.0)).cwiseProduct(s1 + x).sum();

        const Scalar C = s2 - (s1 / (scale + n)).cwiseProduct(s1).sum();

        Scalar ret = -0.5 * d * ln2pi;

        ret -= 0.5 * d * fasterlog(n + 1.0 + scale); // new
        ret += 0.5 * d * fasterlog(n + scale);       // old

        ret += fasterlgamma(a0 + (n + 1.0) * d * 0.5); // new
        ret -= fasterlgamma(a0 + n * d * 0.5);         // old

        ret -= (a0 + (n + 1.0) * d * 0.5) * fasterlog(b0 + C_new * 0.5); // new
        ret += (a0 + n * d * 0.5) * fasterlog(b0 + C * 0.5);             // old

        return ret;
    }

    const vec_type &posterior_mean() const { return mu; }

private:
    template <typename Derived>
    inline multi_gaussian_component_t &
    _add(const Eigen::MatrixBase<Derived> &xx)
    {
        const Derived &x = xx.derived();

#ifdef DEBUG
        ASSERT(x.rows() == s1.rows(), "x1.rows() != s1.rows()");
#endif

        n += 1.0;
        s1 += x;
        s2 += x.cwiseProduct(x).sum();

        posterior_update();

        return *this;
    }

    template <typename Derived>
    inline multi_gaussian_component_t &
    _subtract(const Eigen::MatrixBase<Derived> &xx)
    {
        const Derived &x = xx.derived();

#ifdef DEBUG
        ASSERT(x.rows() == s1.rows(), "x.rows() != s1.rows()");
        ASSERT(n >= (1.0 - 1e-8), "Must have at least one element");
#endif

        n -= 1.0;
        s1 -= x;
        s2 -= x.cwiseProduct(x).sum();

        posterior_update();

#ifdef DEBUG
        ASSERT(n > (-1e-8), "Cannot have negative");
#endif

        return *this;
    }

    ///////////////////////////////////
    // Update variational parameters //
    ///////////////////////////////////

    void posterior_update()
    {
        // prior mean = 0
        mu = s1 / (scale + n); // posterior mean
        mu_prec = tau * (scale + n);
        musq = mu.cwiseProduct(mu).sum();
        musq += d / mu_prec;

        // The expected residual sums of squares
        const Scalar R =
            s2 - 2.0 * (s1.transpose() * mu).sum() + (n + scale) * musq;
        const Scalar a = a0 + (n + 1.0) * d * 0.5;
        const Scalar b = b0 + R * 0.5;

        tau = std::min(a / b, tau_max); // posterior precision
        lntau = fasterdigamma(a) - fasterlog(b);
    }

private:
    const size_t p; // dimensionality
    const Scalar d; // dimensionality
    Scalar n;       // sum_i z_i
    vec_type s1;    // sum_i z_i x_i
    Scalar s2;      // sum_i z_i <x_i,x_i>
    Scalar scale;   // hyper-parameter
    Scalar a0;      // hyper for precision
    Scalar b0;      // hyper for precision
    vec_type mu;    // variational mu
    Scalar mu_prec; //
    Scalar tau;     // variational precision
    Scalar lntau;   // log variational precision
    Scalar musq;    // E[mu^T mu]

    const Scalar ln2pi = fasterlog(2.0 * M_PI);
    const Scalar tau_max = 1e6; // to prevent NaN
};

#endif
