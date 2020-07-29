#include <random>
#include <unordered_map>
#include "eigen_util.hh"

#ifndef MMUTIL_AGGREGATOR_HH_
#define MMUTIL_AGGREGATOR_HH_

struct aggregator_t {

    explicit aggregator_t(const SpMat &_yy, const Mat &_zz)
        : yy(_yy)
        , zz(_zz)
        , D(yy.rows())
        , N(yy.cols())
        , nn(static_cast<Scalar>(N))
        , K(zz.rows())
        , mu(K, D)
        , lambda(N, 1)
        , mu_stat(K, D)
        , lambda_stat(N, 1)
        , ZY(K, D)
        , Ytot(N, 1)
        , temp(K, 1)
        , denomK(K, 1)
        , denomN(N, 1)
        , onesD(D, 1)
    {
        onesD.setOnes();
        lambda.setOnes();
        mu.setZero();
        Ytot = yy.transpose() * onesD;
        ZY = zz * yy.transpose();

        verbose = false;
    }

    const SpMat &yy;
    const Mat &zz;

    const Index D;
    const Index N;
    const Scalar nn;
    const Index K;

    bool verbose;

public:
    void run_gibbs(const Index ngibbs,
                   const Index burnin = 10,
                   const Scalar a0 = 1e-4,
                   const Scalar b0 = 1e-4)
    {
        if (verbose)
            solve_mu(a0, b0);
        for (Index iter = 0; iter < (burnin + ngibbs); ++iter) {
            sample_lambda();
            sample_mu();
            if (iter >= burnin) {
                lambda_stat(lambda);
                mu_stat(mu);
            }
            if (verbose) {
                if (iter >= burnin)
                    std::cerr << "\rGibbs   ";
                else
                    std::cerr << "\rBurn-in ";

                std::cerr << "Iter = " << (iter + 1);
                std::cerr << "\r" << std::flush;
            }
        }
    }

private:
    inline void solve_mu(const Scalar a0 = 1e-4, const Scalar b0 = 1e-4)
    {
        denomK = zz * lambda / nn;

        auto _norm = [a0, b0](const Scalar &a, const Scalar b) {
            return (a + a0) / (b + b0);
        };

        for (Index g = 0; g < D; ++g) {
            mu.col(g) = ZY.col(g).binaryExpr(denomK, _norm);
        }
    }

    inline void sample_mu(const Scalar a0 = 1e-4, const Scalar b0 = 1e-4)
    {
        denomK = zz * lambda / nn;

        auto _sample = [this, a0, b0](const Scalar &a,
                                      const Scalar &b) -> Scalar {
            const Scalar one = 1.;
            std::gamma_distribution<Scalar> gam{ (a + a0), one / (b + b0) };
            return gam(Rng);
        };

        for (Index g = 0; g < D; ++g) {
            mu.col(g) = ZY.col(g).binaryExpr(denomK, _sample);
        }
    }

    inline void solve_lambda(const Scalar a0 = 1e-4, const Scalar b0 = 1e-4)
    {
        auto _norm = [a0, b0](const Scalar &a, const Scalar b) {
            return (a + a0) / (b + b0);
        };

        denomN = zz.transpose() * mu * onesD / nn;
        lambda = Ytot.binaryExpr(denomN, _norm);
    }

    inline void sample_lambda(const Scalar a0 = 1e-4, const Scalar b0 = 1e-4)
    {
        denomN = zz.transpose() * mu * onesD / nn;

        auto _sample = [this, a0, b0](const Scalar &a,
                                      const Scalar &b) -> Scalar {
            const Scalar one = 1.;
            std::gamma_distribution<Scalar> gam{ (a + a0), one / (b + b0) };
            return gam(Rng);
        };

        lambda = Ytot.binaryExpr(denomN, _sample);
    }

private:
    Mat mu;
    Mat lambda;

public:
    running_stat_t<Mat> mu_stat;
    running_stat_t<Mat> lambda_stat;

private:
    Mat ZY;
    Mat Ytot;

    Mat temp;
    Mat denomK;
    Mat denomN;
    Mat onesD;

    std::mt19937 Rng{ std::random_device{}() };
};

#endif
