#include "mmutil.hh"
#include <random>
#include <unordered_map>

#ifndef MMUTIL_POIS_HH_
#define MMUTIL_POIS_HH_

struct poisson_t {

    explicit poisson_t(const Mat _yy,
                       const Mat _zz,
                       const Scalar _a0,
                       const Scalar _b0)
        : yy(_yy)
        , zz(_zz)
        , a0(_a0)
        , b0(_b0)
        , D(yy.rows())
        , N(yy.cols())
        , K(zz.rows())
        , eval_cf(false)
        , rate_opt_op(a0, b0)
        , rate_opt_ln_op(a0, b0)
        , ent_op(a0, b0)
        , mu(K, D)
        , rho(N, 1)
        , rho_cf(N, 1)
        , ln_mu(K, D)
        , ln_rho(N, 1)
        , ln_rho_cf(N, 1)
        , ent_mu(K, D)
        , ent_rho(N, 1)
        , ent_rho_cf(N, 1)
        , ZY(K, D)
        , Ytot(N, 1)
        , Ytot_cf(N, 1)
        , denomK(K, 1)
        , denomN(N, 1)
        , onesD(D, 1)
    {
        onesD.setOnes();
        rho.setOnes();
        ln_rho.setZero();
        mu.setOnes();
        ln_mu.setZero();
        Ytot = yy.transpose() * onesD; // N x 1
        ZY = zz * yy.transpose();      // K x D
        verbose = false;
    }

    explicit poisson_t(const Mat _yy,
                       const Mat _zz,
                       const Mat _yy_cf,
                       const Mat _zz_cf,
                       const Scalar _a0,
                       const Scalar _b0)
        : yy(_yy)
        , zz(_zz)
        , yy_cf(_yy_cf)
        , zz_cf(_zz_cf)
        , a0(_a0)
        , b0(_b0)
        , D(yy.rows())
        , N(yy.cols())
        , K(zz.rows())
        , eval_cf(true)
        , rate_opt_op(a0, b0)
        , rate_opt_ln_op(a0, b0)
        , ent_op(a0, b0)
        , mu(K, D)
        , rho(N, 1)
        , rho_cf(N, 1)
        , ln_mu(K, D)
        , ln_rho(N, 1)
        , ln_rho_cf(N, 1)
        , ent_mu(K, D)
        , ent_rho(N, 1)
        , ent_rho_cf(N, 1)
        , ZY(K, D)
        , Ytot(N, 1)
        , Ytot_cf(N, 1)
        , denomK(K, 1)
        , denomN(N, 1)
        , onesD(D, 1)
    {
        onesD.setOnes();

        rho.setConstant(a0 / b0);
        rho_cf.setConstant(a0 / b0);

        ln_rho.setConstant(fasterdigamma(a0) - fasterlog(b0));
        ln_rho_cf.setConstant(fasterdigamma(a0) - fasterlog(b0));

        mu.setOnes();
        ln_mu.setZero();

        Ytot = yy.transpose() * onesD;       // N x 1
        Ytot_cf = yy_cf.transpose() * onesD; // N x 1

        ZY = zz * yy.transpose();        // K x D
        ZY += zz_cf * yy_cf.transpose(); // K x D

        verbose = false;
    }

    const Mat yy;
    const Mat zz;

    const Mat yy_cf;
    const Mat zz_cf;

    const Scalar a0;
    const Scalar b0;

    const Index D;
    const Index N;
    const Index K;

    const bool eval_cf;

public:
    inline Scalar elbo() const
    {
        Scalar ret = 0.;
        // log-likelihood
        ret += ln_mu.cwiseProduct(ZY).sum();
        ret += ln_rho.cwiseProduct(Ytot).sum();
        ret -= (mu.transpose() * zz * rho).sum();

        // entropy
        ret += ent_mu.sum();
        ret += ent_rho.sum();

        // counterfactual model
        if (eval_cf) {
            ret += ln_rho_cf.cwiseProduct(Ytot_cf).sum();
            ret -= (mu.transpose() * zz_cf * rho_cf).sum();
            ret += ent_rho_cf.sum();
        }
        return ret;
    }

    inline Scalar optimize(const Scalar tol = 1e-4)
    {

        const Scalar denom = static_cast<Scalar>(D * N);

        solve_mu();
        solve_rho();
        Scalar score = elbo() / denom;

        TLOG(score);

        const Index maxIter = 100;

        for (Index iter = 0; iter < maxIter; ++iter) {
            solve_mu();
            solve_rho();
            Scalar _score = elbo() / denom;
            Scalar diff = (score - _score) / (std::abs(score) + tol);

            if (iter > 3 && diff < tol) {
                break;
            }

            score = _score;
            TLOG(score);
        }

        return score;
    }

    inline Mat mu_DK() const { return mu.transpose(); }

    inline Mat rho_N() const { return rho; }

    inline Mat rho_cf_N() const { return rho_cf; }

private:
    inline void solve_mu()
    {
        denomK = zz * rho + zz_cf * rho_cf;

        for (Index g = 0; g < D; ++g) {
            mu.col(g) = ZY.col(g).binaryExpr(denomK, rate_opt_op);
            ln_mu.col(g) = ZY.col(g).binaryExpr(denomK, rate_opt_ln_op);
            ent_mu.col(g) = ZY.col(g).binaryExpr(denomK, ent_op);
        }
    }

    inline void solve_rho()
    {
        // observed model
        denomN = zz.transpose() * (mu * onesD);
        rho = Ytot.binaryExpr(denomN, rate_opt_op);
        ln_rho = Ytot.binaryExpr(denomN, rate_opt_ln_op);
        ent_rho = Ytot.binaryExpr(denomN, ent_op);

        // counterfactual model
        if (eval_cf) {
            denomN = zz_cf.transpose() * (mu * onesD);
            rho_cf = Ytot_cf.binaryExpr(denomN, rate_opt_op);
            ln_rho_cf = Ytot_cf.binaryExpr(denomN, rate_opt_ln_op);
            ent_rho_cf = Ytot_cf.binaryExpr(denomN, ent_op);
        }
    }

public:
    struct rate_opt_op_t {
        explicit rate_opt_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return (a + a0) / (b + b0);
        }
        const Scalar a0, b0;
    };

    struct rate_opt_ln_op_t {
        explicit rate_opt_ln_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return fasterdigamma(a + a0) - fasterlog(b + b0);
        }
        const Scalar a0, b0;
    };

    struct ent_op_t {
        explicit ent_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            const Scalar _a = a + a0;
            const Scalar _b = b + b0;
            Scalar ret = -(_a)*fasterlog(_b);
            ret += fasterlgamma(_a);
            ret -= (_a - 1.) * (fasterdigamma(_a) - fasterlog(_b));
            ret += (_a);
            return ret;
        }
        const Scalar a0, b0;
    };

private:
    rate_opt_op_t rate_opt_op;
    rate_opt_ln_op_t rate_opt_ln_op;
    ent_op_t ent_op;

private:
    Mat mu;
    Mat rho;
    Mat rho_cf;

    Mat ln_mu;
    Mat ln_rho;
    Mat ln_rho_cf;

    Mat ent_mu;
    Mat ent_rho;
    Mat ent_rho_cf;

    Mat ZY;      // combined
    Mat Ytot;    // observed
    Mat Ytot_cf; // counterfactual

    Mat denomK;
    Mat denomN;
    Mat onesD;

    bool verbose;
};

#endif
