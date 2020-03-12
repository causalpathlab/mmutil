#include <random>

#include "mmutil.hh"
#include "svd.hh"
#include "eigen_util.hh"
#include "inference/adam.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_EMBEDDING_HH_
#define MMUTIL_EMBEDDING_HH_

// @param A N x N adjacency matrix
// @param d target dimensionality
template <typename OPTIONS>
inline Mat
train_tsne(const Mat A, const Index d, const OPTIONS &options)
{

    auto kl_op = [](const Scalar &_p, const Scalar &_q) -> Scalar {
        const Scalar eps = 1e-8;
        return _p * fasterlog(_q + eps);
    };

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<Scalar> rnorm{ 0, 1 };

    auto rnorm_jitter_op = [&rnorm, &gen](const Scalar &x) -> Scalar {
        return x + rnorm(gen);
    };

    using grad_adam_t = adam_t<Mat, Scalar>;

    ASSERT(A.rows() == A.cols(), "Symmetric Adj matrix for t-SNE");

    const Index N = A.rows();

    TLOG("Running t-SNE for adjacency matrix A: " << N << " x " << N);

    // Build graph Laplacian to initialize the coordinates
    Mat D = A.rowwise().sum();
    const Scalar tau = std::max(D.mean(), static_cast<Scalar>(1.0));
    D += Mat::Ones(N, 1) * tau;
    const Mat dSqInv = D.cwiseSqrt().cwiseInverse();
    const Mat L = dSqInv.asDiagonal() * A * dSqInv.asDiagonal();

    TLOG("Initialize by SVD");

    Eigen::BDCSVD<Mat> svd;
    svd.compute(L, Eigen::ComputeThinU | Eigen::ComputeThinV);

    TLOG("Setting " << d << " dimensions");

    Mat phi(d, N);
    phi.setZero();

    if (svd.matrixV().cols() > d) { // ignore the first Eigen if possible
        for (Index j = 1; j <= d; ++j) {
            phi.row(j - 1) += svd.matrixV().col(j).transpose();
        }
    } else {
        const Index dd = std::min(d, svd.matrixV().cols());
        for (Index j = 0; j < dd; ++j) {
            phi.row(j) += svd.matrixV().col(j).transpose();
        }
        for (Index j = dd; j < d; ++j) {
            phi.row(j) = phi.row(j).unaryExpr(rnorm_jitter_op) * 0.1;
        }
    }

    TLOG("Start optimizing KL ...");

    Mat q_phi(N, N);
    q_phi.setZero();

    std::vector<std::unique_ptr<Mat>> grad_phi_vec;
    std::vector<std::unique_ptr<grad_adam_t>> adam_phi_vec;
    grad_phi_vec.reserve(N);
    adam_phi_vec.reserve(N);
    for (Index k = 0; k < N; ++k) {
        grad_phi_vec.emplace_back(std::make_unique<Mat>(d, 1));
        adam_phi_vec.emplace_back(
            std::make_unique<grad_adam_t>(0.5, 0.9, d, 1));
    }

    if (options.verbose)
        TLOG("Created gradient matrices");

    Scalar score_old = 0;

    const Index tot_iter = options.embedding_epochs + options.exaggeration;
    progress_bar_t<Index> prog(tot_iter, 1);

    const Scalar T = static_cast<Scalar>(options.embedding_epochs);

    for (Index iter = 0; iter < tot_iter; ++iter) {

        Scalar rate = 1e-2;

        if (iter >= options.exaggeration) {
            Scalar tt = static_cast<Scalar>(iter - options.exaggeration);
            rate *= 1.0 - (tt / T);
        }

        for (Index k = 0; k < N; ++k) {

            // q(k,l) = 1/(1 + norm(phi[k] - phi[l])^2)
            for (Index l = 0; l < N; ++l) {
                if (l == k)
                    continue;

                const Scalar d_lk = (phi.col(k) - phi.col(l)).norm();
                q_phi(l, k) = 1.0 / (1.0 + d_lk * d_lk);
            }
            q_phi.col(k) /= q_phi.col(k).sum();

            Mat &grad_phi = *grad_phi_vec[k].get();
            grad_adam_t &adam_phi = *adam_phi_vec[k].get();

            grad_phi.setZero();
            // d[k,l] = (phi[k] - phi[l])
            // grad[k] = (A[k,l] - q[k,l]) / (1 + d[k,l]^2) * d[k,l]
            for (Index l = 0; l < N; ++l) {
                if (l == k)
                    continue;

                Scalar p_lk = A(l, k);
                if (iter < options.exaggeration)
                    p_lk *= static_cast<Scalar>(N);

                const Scalar d_lk = (phi.col(k) - phi.col(l)).norm();
                const Scalar denom_lk = (1.0 + d_lk * d_lk);
                const Scalar stuff = (p_lk - q_phi(l, k)) / denom_lk;
                grad_phi += stuff * (phi.col(l) - phi.col(k));
            }

            grad_phi -= phi.col(k) * options.l2_penalty; // mild penalty

            phi.col(k) += update_adam(adam_phi, grad_phi) * rate;
        }

        Scalar score = A.binaryExpr(q_phi, kl_op).sum();

        const Scalar score_diff =
            std::abs(score - score_old) / std::abs(score_old + 1.0);

        prog(std::cerr);

        if (iter > options.exaggeration && score_diff < options.tol) {
            TLOG("converged --> iter = " << iter << ", delta = " << score_diff);
            break;
        }

        score_old = score;
    }

    TLOG("Finished the optimization: KL = " << score_old);

    if (options.verbose) {
        TLOG("phi matrix: " << phi.rows() << " x " << phi.cols());
    }

    return phi;
}

template <typename Derived, typename Derived2>
inline Mat
build_smooth_adjacency(Eigen::MatrixBase<Derived> &X,
                       Eigen::MatrixBase<Derived2> &_target)
{

    Mat xx = X.derived();                 // D x N
    normalize_columns(xx);                // D x N
    Derived2 &target = _target.derived(); // N x 1

    const Index nn = xx.cols();
    const Scalar beta_max = 20.;
    Scalar beta = 1e-5;
    auto exp_op_beta = [&beta](const Scalar &_d) -> Scalar {
        return fasterexp(beta * _d);
    };

    auto ln_op = [](const Scalar &x) -> Scalar {
        if (x < 1e-8)
            return 1e-8;
        return fasterlog(x) / fasterlog(2.0);
    };

    Scalar H_denom = fasterlog(static_cast<Scalar>(nn)) / fasterlog(2.0);

    Vec pr_i(nn);
    Vec dist_i(nn);
    Mat pp(nn, nn);

    for (Index i = 0; i < nn; ++i) {

        beta = 1.0;

        Scalar target_i = target(i);

        for (Index _iter = 0; _iter < 100; _iter++) {
            pr_i.setZero();

            pr_i = (xx.transpose() * xx.col(i)).unaryExpr(exp_op_beta);
            pr_i(i) = 0;
            pr_i /= pr_i.sum() + 1e-8;

            Scalar H = -((pr_i.unaryExpr(ln_op)).cwiseProduct(pr_i)).sum();
            H /= H_denom;

            if (H < target_i) {
                beta *= .9;
            } else {
                beta *= 1.1;
            }

            if (beta > beta_max)
                break;
        }

        pp.col(i) = pr_i;
        // std::cout << beta << std::endl;
    }

    Mat ret = (pp + pp.transpose()) * 0.5 / static_cast<Scalar>(nn);

    return ret;
}

////////////////////////////////////////////////////////////
// Embedding data points based on soft-clustering results //
// 							  //
// X  : N x D data matrix				  //
// Pr : N x K assignment probability			  //
////////////////////////////////////////////////////////////

template <typename Derived, typename Derived2, typename OPTIONS>
std::tuple<Mat, Mat>
train_embedding_by_cluster(const Eigen::MatrixBase<Derived> &_x,
                           const Eigen::MatrixBase<Derived2> &_pr,
                           const OPTIONS &options)
{

    const Derived &X = _x.derived();
    const Derived2 &Pr = _pr.derived();

    const Index d = options.embedding_dim;
    const Index K = Pr.cols();
    const Index N = Pr.rows();

    ASSERT(X.rows() == Pr.rows(), "X and Pr should have the same # rows")

    ///////////////////////////////////////////////
    // smooth probability with target perplexity //
    ///////////////////////////////////////////////

    Vec nsize = Pr.colwise().sum().transpose();                       // K x 1
    Mat C = (X.transpose() * Pr) * nsize.cwiseInverse().asDiagonal(); // D x K
    Vec pr_size = nsize / nsize.sum();
    Mat p_phi = build_smooth_adjacency(C, pr_size);
    Mat phi = train_tsne(p_phi, d, options);

    // TLOG("phi:\n" << phi);

    Eigen::BDCSVD<Mat> svd;
    svd.compute(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Scalar tau = svd.singularValues().mean();
    tau = std::sqrt(tau);

    auto safe_inverse = [&tau](const Scalar &x) -> Scalar {
        return 1.0 / (x + tau);
    };

    TLOG("Projecting data into " << d << " dimensions");

    Vec denom = svd.singularValues().unaryExpr(safe_inverse);
    Mat proj = svd.matrixV() * denom.asDiagonal() * svd.matrixU().transpose();
    Mat ww = phi * proj;

    Mat yy = ww * X.transpose(); // d x N
    yy.transposeInPlace();       // N x d
    phi.transposeInPlace();      // K x d

    TLOG("done");

    return std::make_tuple(yy, phi);
}

template <typename OPTIONS>
inline std::tuple<Mat, Mat>
embed_by_centroid(const Mat &X,                   //
                  const std::vector<Index> &memb, //
                  const OPTIONS &options)
{
    const Index K = *std::max_element(memb.begin(), memb.end()) + 1;
    const Index D = X.rows();
    const Index N = X.cols();

    if (K < 2) {
        TLOG("#clusters < 2");
        return std::make_tuple(Mat{}, Mat{});
    }

    const Scalar kk = K;
    const Scalar min_pr = 1.0 / (kk - 1.0) / kk;
    const Scalar max_pr = 1.0 - (kk - 1.0) * min_pr;

    Mat Pr(N, K);
    Pr.setConstant(min_pr);
    for (Index j = 0; j < N; ++j) {
        const Index k = memb.at(j);
        if (k >= 0) {
            Pr(j, k) = max_pr;
        } else {
            for (Index l = 0; l < K; ++l)
                Pr(j, l) = 1.0 / kk;
        }
    }

    return train_embedding_by_cluster(X, Pr, options);
}

#endif
