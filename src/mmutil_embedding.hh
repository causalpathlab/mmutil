#include <random>

#include "mmutil.hh"
#include "svd.hh"
#include "eigen_util.hh"
#include "inference/adam.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_EMBEDDING_HH_
#define MMUTIL_EMBEDDING_HH_

////////////////////////////////////////////////////////////////
// Embedding soft-clustering results
//
// Pr : N x K assignment probability
// X  : N x D data matrix
//
template <typename T>
std::tuple<Mat, Mat>
run_cluster_embedding(const Mat &Pr, const Mat &X, const T &options)
{

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

    Mat prob = (cc.transpose() * xt) / sig2 * 0.5; // K x N

    Vec mass_K(K);
    Vec pr_K(K);

    for (Index i = 0; i < N; ++i) {
        mass_K = prob.col(i);
        normalized_exp(mass_K, pr_K);
        prob.col(i) = pr_K;
    }

    TLOG("Constructed a smooth probability matrix");

    ///////////////////////////
    // shatter the centroids //
    ///////////////////////////
    using grad_adam_t = adam_t<Mat, Scalar>;

    Mat p_phi = (cc.transpose() * cc).unaryExpr(exp_op);
    for (Index k = 0; k < p_phi.cols(); ++k) {
        p_phi.col(k) /= p_phi.col(k).sum();
    }

    // Embedded centroid matrix [d x K]

    Eigen::BDCSVD<Mat> svd;
    svd.compute(cc, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Mat phi = standardize(svd.matrixV().topRows(d));
    phi += Mat::Zero(d, K).unaryExpr(rnorm_jitter_op) * 0.01;
    Mat q_phi(K, K);

    std::vector<std::shared_ptr<Mat>> grad_phi_vec;
    std::vector<std::shared_ptr<grad_adam_t>> adam_phi_vec;
    for (Index k = 0; k < K; ++k) {
        grad_phi_vec.push_back(std::make_shared<Mat>(d, 1));
        adam_phi_vec.push_back(std::make_shared<grad_adam_t>(0.5, 0.9, d, 1));
    }

    for (Index t = 0; t < options.embedding_epochs; ++t) {

        const Scalar rate = options.embedding_rate;

        // q(k,l) = 1/(1 + norm(phi[k] - phi[l])^2)

        for (Index k = 0; k < K; ++k) {
            for (Index l = 0; l < K; ++l) {
                const Scalar d_lk = (phi.col(k) - phi.col(l)).norm();
                q_phi(l, k) = 1.0 / (1.0 + d_lk * d_lk);
            }
            q_phi.col(k) /= q_phi.col(k).sum();
        }

        for (Index k = 0; k < K; ++k) {

            Mat &grad_phi = *grad_phi_vec[k].get();
            grad_adam_t &adam_phi = *adam_phi_vec[k].get();

            const Scalar denom = static_cast<Scalar>(phi.cols());
            grad_phi.setZero();
            // d[k,l] = (phi[k] - phi[l])
            // grad[k] = (A[k,l] - q[k,l]) / (1 + d[k,l]^2) * d[k,l]
            for (Index l = 0; l < K; ++l) {
                if (l == k)
                    continue;
                const Scalar d_lk = (phi.col(k) - phi.col(l)).norm();
                const Scalar denom_lk = (1.0 + d_lk * d_lk);
                const Scalar stuff = (p_phi(l, k) - q_phi(l, k)) / denom_lk;
                grad_phi += stuff * (phi.col(k) - phi.col(l)) / denom;
            }

            grad_phi -= phi.col(k) * 0.1; // regularization

            phi.col(k) += update_adam(adam_phi, grad_phi) * rate;
        }
    }

    std::cerr << std::endl;
    std::cerr << phi << std::endl;
    std::cerr << std::endl;

    // Mat grad_rank(d, 1);
    // grad_adam_t adam_rank(0.9, 0.99, d, 1);
    // Mat phi = Mat::Zero(d, K);
    // phi = phi.unaryExpr(rnorm_jitter_op);
    // const Index n_shatter = 100;
    // for (Index t = 0; t < n_shatter; ++t) {
    //     const Scalar rate = 0.01;
    //     const Scalar denom = static_cast<Scalar>(phi.cols());
    //     for (Index k = 0; k < K; ++k) {
    //         grad_rank.setZero();
    //         for (Index l = 0; l < K; ++l) {
    //             if (k == l)
    //                 continue;
    //             const Scalar dist = (phi.col(k) - phi.col(l)).norm();
    //             const Scalar qq = sigmoid_op(-0.5 * dist * dist);
    //             grad_rank -= qq * phi.col(l);
    //         }
    //         grad_rank /= denom;
    //         phi.col(k) += update_adam(adam_rank, grad_rank) * rate;
    //     }
    // }
    // phi.transposeInPlace();

    ////////////////////
    // random seeding //
    ////////////////////

    if (options.verbose)
        TLOG("Random seeding latent coordinates ...");

    // d x N latent coordinate
    Mat yy = Mat::Zero(d, N).unaryExpr(rnorm_jitter_op) * 0.01;

    std::vector<std::shared_ptr<Mat>> grad_y_vec;
    std::vector<std::shared_ptr<grad_adam_t>> adam_y_vec;
    for (Index i = 0; i < N; ++i) {
        grad_y_vec.push_back(std::make_shared<Mat>(d, 1));
        adam_y_vec.push_back(std::make_shared<grad_adam_t>(0.5, 0.9, d, 1));
    }

    ////////////////////////////////////////////////////////////
    // Approximating assignment probability by t-distribution //
    // delta = (y[j, d] - phi[k, d])			      //
    // q[j, k] = 1 / (1 + norm(delta)^2)		      //
    ////////////////////////////////////////////////////////////

    Vec qq(K); // K x 1 temporary

    progress_bar_t<Index> prog(options.embedding_epochs, 1);

    Scalar score_old = 0;
    for (Index iter = 0; iter < options.embedding_epochs; ++iter) {

        const Scalar rate = options.embedding_rate;

        for (Index i = 0; i < N; ++i) {

            // q = 1/(1 + distance^2)
            for (Index k = 0; k < K; ++k) {
                const Scalar d_ik = (yy.col(i) - phi.col(k)).norm();
                qq(k) = 1.0 / (1.0 + d_ik * d_ik);
            }
            qq /= qq.sum();
        }

        for (Index i = 0; i < N; ++i) {

            Mat &grad_y = *grad_y_vec[i].get();
            grad_adam_t &adam_y = *adam_y_vec[i].get();

            grad_y.setZero();

            // d[i,k] = norm(y[,i] - phi[,k])
            // grad = sum_k (p(i,k) - q(i,k))/(1 + d[i,k]^2) * (y[,i] - phi[,k])
            for (Index k = 0; k < K; ++k) {
                const Scalar d_ik = (yy.col(i) - phi.col(k)).norm();
                const Scalar pq_ik = prob(k, i) - qq(k);
                const Scalar stuff = pq_ik / (1.0 + d_ik * d_ik);
                grad_y += (yy.col(i) - phi.col(k)) * stuff;
            }

            grad_y -= yy.col(i) * 0.1; // regularization

            yy.col(i) += update_adam(adam_y, grad_y) * rate;
        }

        prog.update();

        prog(std::cerr);

        // const Scalar score =
        //     prob.binaryExpr(Qr, kl_op).sum() / static_cast<Scalar>(N);
        // const Scalar score_diff = std::abs(score - score_old);

        // if (iter >= 10 && score_diff < options.tol) {
        //     TLOG("converged");
        //     break;
        // }

        // if (!options.verbose) {
        //     prog(std::cerr);
        // } else {
        //     TLOG("[" << iter << "] [" << score << "]");
        // }
    }

    phi.transposeInPlace();
    yy.transposeInPlace();

    return std::make_tuple(yy, phi);
}

template <typename OPTIONS>
inline Mat
embed_by_centroid(const Mat &X,                   //
                  const std::vector<Index> &memb, //
                  const OPTIONS &options)
{
    const Index kk = *std::max_element(memb.begin(), memb.end()) + 1;
    const Index D = X.rows();
    const Index N = X.cols();

    if (kk < 1) {
        TLOG("There is no centroid");
        return Mat{};
    }

    // TODO1:
    // Calibrate distance from each centroid and estimate the
    // probability of assignments.

    // const Index n_rand_clust = 3; // extra clusters for randomness

    // Mat _cc(D, kk + n_rand_clust); // take the means
    // Vec _nn(kk + n_rand_clust);    // denominator
    // _cc.setZero();
    // _nn.setOnes();

    // for (Index j = 0; j < N; ++j) {
    //     const Index k = memb.at(j); //
    //     if (k >= 0) {               // ignore unassigned
    //         _cc.col(k) += X.col(j); // accumulate
    //         _nn(k) += 1.0;          //
    //     }
    // }

    // /////////////////////////////
    // // create random clusters  //
    // /////////////////////////////

    // std::random_device rd;
    // std::mt19937 rgen(rd());

    // std::vector<Index> rand_i(N);
    // std::iota(rand_i.begin(), rand_i.end(), 0);
    // std::shuffle(rand_i.begin(), rand_i.end(), rgen);

    // const Index n_rand = (1 + N / kk) * n_rand_clust;

    // if (options.verbose) {
    //     TLOG("Generating some cluster of random points");
    // }

    // for (Index r = 0; r < n_rand; ++r) {
    //     const Index j = r % N;
    //     const Index k = kk + (r % n_rand_clust);
    //     _cc.col(k) += X.col(j);
    //     _nn(k) += 1.0;
    // }

    // ////////////////////////////
    // // normalize the centroid //
    // ////////////////////////////

    // Mat C = _cc * (_nn.cwiseInverse().asDiagonal()); // D x K

    // const Index nepochs = options.embedding_epochs;
    const Index dd = std::min(options.embedding_dim, kk);
    Mat Y = Mat::Random(dd, N);

    // const Index rank = dd + 1;

    // if (options.verbose) {
    //     TLOG("Start embedding, epochs = " << nepochs << ", rank = " << rank);
    // }

    // Mat CC = scale_by_degree(C, options.tau);
    // Mat XX = scale_by_degree(X, options.tau);

    // //////////////////////////////////////////////////////////
    // // Smooth probability of assignment by von Mises-Fisher //
    // //////////////////////////////////////////////////////////

    // Mat P(CC.cols(), XX.cols());
    // Vec mass(CC.cols());
    // Vec pr_i(CC.cols());

    // Mat xx_unit = XX;
    // Mat cc_unit = CC;

    // // MUST: Quick estimation of concentration parameters
    // // MUST: make the probability sharpened

    // normalize_columns(xx_unit);
    // normalize_columns(cc_unit);

    // for (Index i = 0; i < N; ++i) {

    //     for (Index k = 0; k < cc_unit.cols(); ++k) {
    //         mass(k) = cc_unit.col(k).transpose() * xx_unit.col(i);
    //     }

    //     normalized_exp(mass, pr_i);
    //     P.col(i) = pr_i;
    // }

    // Mat phi = Mat::Random(dd, CC.cols());
    // Mat Y = Mat::Random(dd, XX.cols());

    // if (options.verbose) {
    //     TLOG("Built initial embedding: " << Y.rows() << " x " << Y.cols());
    // }

    // ///////////////////////////
    // // shatter the centroids //
    // ///////////////////////////

    // Vec diff_rank(dd);
    // Vec grad_rank(dd);

    // auto _sigmoid_op = [](const Scalar x) -> Scalar {
    //     const Scalar _pmin = 0.01;
    //     const Scalar _pmax = 0.99;
    //     return _sigmoid(x, _pmin, _pmax);
    // };

    // const Index n_shatter = 100;
    // normalize_columns(phi);

    // for (Index t = 0; t < n_shatter; ++t) {

    //     const Scalar rate =
    //         1.0 - static_cast<Scalar>(t) / static_cast<Scalar>(n_shatter);

    //     const Scalar denom = static_cast<Scalar>(phi.cols());

    //     for (Index k = 0; k < kk; ++k) {
    //         grad_rank.setZero();
    //         for (Index l = 0; l < (kk + n_rand_clust); ++l) {
    //             if (k == l)
    //                 continue;

    //             const Scalar dist = phi.col(k).transpose() * phi.col(l);
    //             const Scalar qq = _sigmoid_op(dist);

    //             grad_rank -= qq * phi.col(l);
    //         }
    //         phi.col(k) += rate * grad_rank / denom;
    //     }

    //     normalize_columns(phi);
    // }

    // //////////////////////////////////////////
    // // optimization by parametric embedding //
    // //////////////////////////////////////////

    // Mat Q(CC.cols(), XX.cols());
    // Vec qq_i(Q.rows());

    // auto update_Q = [&]() {
    //     for (Index i = 0; i < N; ++i) {
    //         for (Index k = 0; k < CC.cols(); ++k) {
    //             diff_rank = (Y.col(i) - phi.col(k));
    //             const Scalar dist = diff_rank.cwiseProduct(diff_rank).sum();
    //             mass(k) = -0.5 * dist;
    //         }
    //         normalized_exp(mass, qq_i);
    //         Q.col(i) = qq_i;
    //     }
    // };

    // const Scalar reg = 1e-2;

    // auto update_Y = [&](const Scalar rate = 1e-2) {
    //     for (Index i = 0; i < N; ++i) {
    //         grad_rank = (phi.colwise() - Y.col(i)) * (P.col(i) - Q.col(i));
    //         Y.col(i) -= (grad_rank + reg * Y.col(i)) * rate;
    //     }
    // };

    // const Index n_estep = 100;

    // for (Index t = 0; t < n_estep; ++t) {
    //     const Scalar rate =
    //         1.0 - static_cast<Scalar>(t) / static_cast<Scalar>(n_estep);
    //     update_Q();
    //     update_Y(rate);
    // }

    // // TODO

    // // phi = dd x K

    // // grad_rank.setZero();
    // // for (Index i = 0; i < N; ++i) {
    // //   grad_rank +=
    // // 	(phi.colwise() - Y.col(i)).transpose() * (Q.col(i) - P.col(i));
    // // }

    // // phi -= (grad_rank + reg * Y.col(i)) * rate / static_cast<Scalar>(N);

    // // //////////////////////////////////////////
    // // // Move data points in the latent space //
    // // //////////////////////////////////////////
    // // Vec ln_pr(kk + n_rand_clust);

    // // for (Index t = 0; t < nepochs; ++t) {
    // //     const Scalar rate =
    // //         1.0 - static_cast<Scalar>(t) / static_cast<Scalar>(nepochs);

    // //     for (Index j = 0; j < N; ++j) {
    // //         Index k = memb.at(j);
    // //         if (k < 0) {
    // //             ln_pr = Cd.transpose() * Y.col(j);
    // //             ln_pr.maxCoeff(&k);
    // //         }

    // //         grad_rank.setZero();

    // //         // Gradient to separate clusters
    // //         {
    // //             const Scalar p_obs = P(k, j);

    // //             diff_rank = Cd.col(k) - Y.col(j);
    // //             const Scalar dist =
    // diff_rank.cwiseProduct(diff_rank).sum();
    // //             const Scalar pp = _sigmoid_op(-0.5 * dist);

    // //             grad_rank -= (p_obs - pp) * diff_rank;
    // //         }

    // //         for (Index l = 0; l < (kk + n_rand_clust); ++l) {
    // //             if (l == k)
    // //                 continue;

    // //             const Scalar p_obs = P(l, j);

    // //             diff_rank = Cd.col(l) - Y.col(j);
    // //             const Scalar dist =
    // diff_rank.cwiseProduct(diff_rank).sum();
    // //             const Scalar qq = _sigmoid_op(-0.5 * dist);

    // //             grad_rank -= (p_obs - qq) * diff_rank;
    // //         }

    // //         Y.col(j) +=
    // //             rate * grad_rank / static_cast<Scalar>(kk + n_rand_clust);
    // //     }
    // // }

    // // if (options.verbose) {
    // //     TLOG("Refinement: [" << Y.rows() << " x " << Y.cols() << "]");
    // // }

    Y.transposeInPlace();
    return Y;
}

#endif
