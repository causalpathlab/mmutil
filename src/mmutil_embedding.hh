#include <random>

#include "mmutil.hh"
#include "svd.hh"
#include "eigen_util.hh"
#include "inference/adam.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_EMBEDDING_HH_
#define MMUTIL_EMBEDDING_HH_

template <typename OPTIONS>
inline Mat
train_tsne(const Mat A, const Index d, const OPTIONS &options)
{

    auto kl_op = [](const Scalar &_p, const Scalar &_q) -> Scalar {
        const Scalar eps = 1e-8;
        return _p * fasterlog(_q + eps);
    };

    using grad_adam_t = adam_t<Mat, Scalar>;

    ASSERT(A.rows() == A.cols(), "Symmetric Adj matrix for t-SNE");

    const Index N = A.rows();

    TLOG("Running t-SNE for adjacency matrix A: " << N << " x " << N);

    // column-wise normalize

    Mat p_phi = A;

    for (Index k = 0; k < N; ++k) {
        p_phi(k, k) = 0.;
        const Scalar denom_k = p_phi.col(k).sum();
        const Scalar denom_min = 1e-8;
        p_phi.col(k) /= std::max(denom_k, denom_min);
    }

    // Build graph Laplacian to initialize the coordinates
    Mat D = A.rowwise().sum();
    const Scalar tau = std::max(D.mean(), static_cast<Scalar>(1.0));
    D += Mat::Ones(N, 1) * tau;
    const Mat dSqInv = D.cwiseSqrt().cwiseInverse();
    const Mat L = dSqInv.asDiagonal() * A * dSqInv.asDiagonal();

    TLOG("Initialize by SVD");

    Eigen::BDCSVD<Mat> svd;
    svd.compute(L, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Mat phi(d, N);
    phi.setZero();

    if (svd.matrixV().cols() > d) { // ignore the first Eigen if possible
        for (Index j = 1; j <= d; ++j) {
            phi.row(j) += svd.matrixV().col(j - 1).transpose();
        }
    } else {
        for (Index j = 0; j < d; ++j) {
            phi.row(j) += svd.matrixV().col(j).transpose();
        }
    }

    Mat q_phi(N, N);
    q_phi.setZero();

    std::vector<std::unique_ptr<Mat>> grad_phi_vec;
    std::vector<std::unique_ptr<grad_adam_t>> adam_phi_vec;
    for (Index k = 0; k < N; ++k) {
        grad_phi_vec.push_back(std::make_unique<Mat>(d, 1));
        adam_phi_vec.push_back(std::make_unique<grad_adam_t>(0.5, 0.9, d, 1));
    }

    Scalar score_old = 0;

    const Scalar T = static_cast<Scalar>(options.embedding_epochs);

    for (Index iter = 0; iter < options.embedding_epochs; ++iter) {

        const Scalar tt = static_cast<Scalar>(iter);
        const Scalar rate = (1.0 - tt / T) * 0.1;

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

                const Scalar d_lk = (phi.col(k) - phi.col(l)).norm();
                const Scalar denom_lk = (1.0 + d_lk * d_lk);
                const Scalar stuff = (p_phi(l, k) - q_phi(l, k)) / denom_lk;
                grad_phi += stuff * (phi.col(l) - phi.col(k));
            }

            grad_phi -= phi.col(k) * 1e-2; // mild penalty

            phi.col(k) += update_adam(adam_phi, grad_phi) * rate;
        }

        Scalar score = p_phi.binaryExpr(q_phi, kl_op).sum();

        const Scalar score_diff =
            std::abs(score - score_old) / std::abs(score_old + 1e-4);

        if (score_diff < options.tol) {
            TLOG("converged --> iter = " << iter << ", delta = " << score_diff);
            break;
        }

        score_old = score;
    }

    if (options.verbose) {
        TLOG("Make sure these two probabilities more or less match");

        std::cerr << "p:" << std::endl;
        std::cerr << p_phi << std::endl;
        std::cerr << "vs." << std::endl;

        std::cerr << "q:" << std::endl;
        std::cerr << q_phi << std::endl;
        std::cerr << std::endl;
    }

    TLOG("phi matrix: " << phi.rows() << " x " << phi.cols());

    return phi;
}

////////////////////////////////////////////////////////////////
// Embedding soft-clustering results
//
// Pr : N x K assignment probability
// X  : N x D data matrix
//
template <typename Derived, typename OPTIONS>
std::tuple<Mat, Mat>
run_cluster_embedding(const Eigen::MatrixBase<Derived> &_pr,
                      const Mat &X,
                      const OPTIONS &options)
{

    const Derived &Pr = _pr.derived();

    const Index d = options.embedding_dim;
    const Index K = Pr.cols();
    const Index N = Pr.rows();
    const Scalar sig2 = options.sig2;

    //////////////////////
    // simple operators //
    //////////////////////

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<Scalar> rnorm{ 0, 1 };

    auto rnorm_jitter_op = [&rnorm, &gen](const Scalar &x) -> Scalar {
        return x + rnorm(gen);
    };

    auto exp_op = [&sig2](const Scalar &x) -> Scalar {
        return fasterexp(x / sig2);
    };

    auto t_op = [](const Scalar &x) -> Scalar {
        const Scalar one = 1.0;
        return one / (one + x * x);
    };

    auto t_grad_op = [](const Scalar &_g, const Scalar &_d) -> Scalar {
        const Scalar one = 1.0;
        return _g / (one + _d * _d);
    };

    auto kl_op = [](const Scalar &_p, const Scalar &_q) -> Scalar {
        const Scalar eps = 1e-8;
        return _p * fasterlog(_q + eps);
    };

    auto sigmoid_op = [](const Scalar x) -> Scalar {
        const Scalar _pmin = 0.01;
        const Scalar _pmax = 0.99;
        return _sigmoid(x, _pmin, _pmax);
    };

    ASSERT(X.rows() == Pr.rows(), "X and Pr should have the same # rows")

    //////////////////////////////////////////
    // smooth probability (no extreme zero) //
    //////////////////////////////////////////

    Vec nsize = Pr.colwise().sum().transpose();                       // K x 1
    Mat C = (X.transpose() * Pr) * nsize.cwiseInverse().asDiagonal(); // D x K

    Mat xt = X.transpose(); // D x N
    Mat cc = C;             // D x K

    normalize_columns(xt); // D x N
    normalize_columns(cc); // D x K

    Mat prob = (cc.transpose() * xt) / sig2; // K x N

    Vec mass_K(K);
    Vec pr_K(K);

    for (Index i = 0; i < N; ++i) {
        mass_K = prob.col(i);
        normalized_exp(mass_K, pr_K);
        prob.col(i) = pr_K;
    }

    TLOG("Constructed a smooth probability matrix");

    using grad_adam_t = adam_t<Mat, Scalar>;

    //////////////////////////////////////
    // Embedded centroid matrix [d x K] //
    //////////////////////////////////////

    Mat p_phi = prob * prob.transpose() / static_cast<Scalar>(N);
    Mat phi = train_tsne(p_phi, d, options);

    TLOG("Finished embedding centroid by t-SNE");

    ////////////////////
    // random seeding //
    ////////////////////

    // TLOG("Pr : " << Pr.rows() << " x " << Pr.cols());
    // TLOG("phi : " << phi.rows() << " x " << phi.cols());

    // d x N latent coordinate
    Mat yy = Pr * phi.transpose();
    yy.transposeInPlace();
    yy *= 0.9;
    yy += Mat::Zero(d, N).unaryExpr(rnorm_jitter_op) * 0.1;

    if (options.verbose) {
        TLOG("Random seeding latent coordinates based on");
        TLOG("the prob matrix: " << Pr.rows() << " x " << Pr.cols());
    }

    std::vector<std::unique_ptr<Mat>> grad_y_vec;
    std::vector<std::unique_ptr<grad_adam_t>> adam_y_vec;
    for (Index i = 0; i < N; ++i) {
        grad_y_vec.push_back(std::make_unique<Mat>(d, 1));
        adam_y_vec.push_back(std::make_unique<grad_adam_t>(0.5, 0.9, d, 1));
    }

    if (options.verbose)
        TLOG("Created gradient matrices");

    ////////////////////////////////////////////////////////////
    // Approximating assignment probability by t-distribution //
    // delta = (y[j, d] - phi[k, d])			      //
    // q[j, k] = 1 / (1 + norm(delta)^2)		      //
    ////////////////////////////////////////////////////////////

    Vec qq(K); // K x 1 temporary
    Vec pp(K); // K x 1 temporary

    progress_bar_t<Index> prog(options.embedding_epochs, 1);

    std::vector<Index> indexes(N);
    std::iota(indexes.begin(), indexes.end(), 0);

    const Scalar T = options.embedding_epochs;
    Scalar score_old = 0;
    for (Index iter = 0; iter < options.embedding_epochs; ++iter) {

        const Scalar tt = static_cast<Scalar>(iter);
        const Scalar rate = (1.0 - tt / T) * 0.1;

        Scalar score = 0;

        std::shuffle(indexes.begin(), indexes.end(), gen);

        for (Index j = 0; j < N; ++j) {

            const Index i = indexes.at(j);

            pp = prob.col(i);
            qq.setZero();

            // q = 1/(1 + distance^2)
            for (Index k = 0; k < K; ++k) {
                const Scalar d_ik = (yy.col(i) - phi.col(k)).norm();
                qq(k) += 1.0 / (1.0 + d_ik * d_ik);
            }
            qq /= qq.sum();

            score += pp.binaryExpr(qq, kl_op).sum();

            // d[i,k] = norm(y[,i] - phi[,k])
            // grad = sum_k (p(i,k) - q(i,k))/(1 + d[i,k]^2) * (y[,i] - phi[,k])

            Mat &grad_y = *grad_y_vec[i].get();
            grad_adam_t &adam_y = *adam_y_vec[i].get();

            grad_y.setZero();

            for (Index k = 0; k < K; ++k) {
                const Scalar d_ik = (yy.col(i) - phi.col(k)).norm();
                const Scalar pq_ik = prob(k, i) - qq(k);
                const Scalar stuff = pq_ik / (1.0 + d_ik * d_ik);
                grad_y += (phi.col(k) - yy.col(i)) * stuff;
            }

            grad_y -= yy.col(i) * options.l2_penalty;

            yy.col(i) += update_adam(adam_y, grad_y) * rate;
        }

        prog.update();

        const Scalar score_diff =
            std::abs(score - score_old) / std::abs(score_old + 1e-4);

        if (score_diff < options.tol) {
            TLOG("converged --> iter = " << iter << ", delta = " << score_diff);
            break;
        }

        score_old = score;

        if (!options.verbose) {
            prog(std::cerr);
        } else {
            TLOG("[" << iter << "] [" << score << "]");
        }
    }

    phi.transposeInPlace(); // K x d
    yy.transposeInPlace();  // N x d

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

    return run_cluster_embedding(Pr, X, options);
}

#endif
